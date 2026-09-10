"""
LLM provider abstraction layer.

Supports two backends:
  - "mock"   → deterministic responses, no API key needed (default)
  - "openai" → real OpenAI-compatible API (set OPENAI_API_KEY)

Swap providers by changing LLM_PROVIDER in .env without touching app logic.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from copilot.config import ModelConfig


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str
    confidence: float   # 0–1, estimated; real APIs don't expose this directly


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rough_token_count(text: str) -> int:
    """Approximate token count: ~4 chars per token."""
    return max(1, len(text) // 4)


# ─── Mock Provider ────────────────────────────────────────────────────────────

_MOCK_RESPONSES_BY_TIER = {
    "small": (
        "Based on available information: {summary}. "
        "For more detail, a deeper analysis may be needed."
    ),
    "medium": (
        "Here is a focused answer: {summary}\n\n"
        "Key points to consider:\n"
        "• The context indicates {first_fact}\n"
        "• Additionally, {second_fact}\n\n"
        "Let me know if you need further clarification."
    ),
    "large": (
        "Comprehensive analysis:\n\n"
        "{summary}\n\n"
        "Detailed breakdown:\n"
        "1. {first_fact}\n"
        "2. {second_fact}\n"
        "3. This analysis accounts for multiple factors including context, "
        "historical patterns, and relevant constraints.\n\n"
        "Confidence assessment: The available context strongly supports this "
        "conclusion. If you have additional specifics, I can refine further."
    ),
}

_CONFIDENCE_BY_TIER = {"small": 0.62, "medium": 0.78, "large": 0.91}


def _extract_sentences(text: str, n: int = 3) -> List[str]:
    """Pull the first n sentences from text."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()][:n]


def _mock_respond(
    messages: List[Message],
    model: ModelConfig,
    context_chunks: Optional[List[str]] = None,
) -> LLMResponse:
    t0 = time.perf_counter()

    # Build a digest so the same input → same mock output (deterministic)
    digest = hashlib.md5(
        "\n".join(f"{m.role}:{m.content}" for m in messages).encode()
    ).hexdigest()[:8]

    # Derive a "summary" from context or the last user message
    context_text = " ".join(context_chunks or []) if context_chunks else ""
    user_text = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )
    source = context_text if context_text else user_text

    sentences = _extract_sentences(source) or ["No specific context available."]
    summary = sentences[0]
    first_fact = sentences[1] if len(sentences) > 1 else "context is limited"
    second_fact = sentences[2] if len(sentences) > 2 else "further research advised"

    template = _MOCK_RESPONSES_BY_TIER[model.tier]
    content = template.format(
        summary=summary, first_fact=first_fact, second_fact=second_fact
    )
    content += f"\n\n*(mock/{model.tier} · id:{digest})*"

    # Simulate latency: small=fast, large=slow
    latency_factor = {"small": 0.08, "medium": 0.22, "large": 0.55}[model.tier]
    elapsed = (time.perf_counter() - t0 + latency_factor) * 1000  # ms

    return LLMResponse(
        content=content,
        input_tokens=_rough_token_count(" ".join(m.content for m in messages)),
        output_tokens=_rough_token_count(content),
        latency_ms=elapsed,
        model_name=f"mock-{model.tier}",
        confidence=_CONFIDENCE_BY_TIER[model.tier],
    )


# ─── OpenAI Provider ─────────────────────────────────────────────────────────

def _openai_respond(
    messages: List[Message],
    model: ModelConfig,
    api_key: Optional[str],
    base_url: Optional[str],
) -> LLMResponse:
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from exc

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model.name,
        messages=[{"role": m.role, "content": m.content} for m in messages],
        max_tokens=model.max_tokens,
        temperature=0.3,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    choice = response.choices[0]
    content = choice.message.content or ""
    usage = response.usage

    # OpenAI doesn't expose confidence; estimate from finish_reason
    confidence = 0.85 if choice.finish_reason == "stop" else 0.55

    return LLMResponse(
        content=content,
        input_tokens=usage.prompt_tokens if usage else _rough_token_count(
            " ".join(m.content for m in messages)
        ),
        output_tokens=usage.completion_tokens if usage else _rough_token_count(content),
        latency_ms=elapsed,
        model_name=model.name,
        confidence=confidence,
    )


# ─── Public interface ─────────────────────────────────────────────────────────

class LLMProvider:
    """Thin wrapper that dispatches to the configured backend."""

    def __init__(
        self,
        llm_provider: str = "mock",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.base_url = base_url

    def complete(
        self,
        messages: List[Message],
        model: ModelConfig,
        context_chunks: Optional[List[str]] = None,
    ) -> LLMResponse:
        if self.llm_provider == "openai":
            return _openai_respond(messages, model, self.api_key, self.base_url)
        return _mock_respond(messages, model, context_chunks)
