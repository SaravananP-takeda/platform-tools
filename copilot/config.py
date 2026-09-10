"""
Configuration for the Enterprise Copilot.
All settings are environment-driven with sensible defaults so the app runs
without any credentials (mock provider) out of the box.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Describes a model tier: name, pricing, and provider."""

    name: str
    tier: str  # "small" | "medium" | "large"
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD
    max_tokens: int
    provider: str  # "mock" | "openai"


@dataclass
class Config:
    # ── Provider ──────────────────────────────────────────────────────────────
    llm_provider: str  # "mock" | "openai"
    openai_api_key: Optional[str]
    openai_base_url: Optional[str]

    # ── Model tiers ───────────────────────────────────────────────────────────
    small_model: ModelConfig
    medium_model: ModelConfig
    large_model: ModelConfig

    # ── Cache ─────────────────────────────────────────────────────────────────
    exact_cache_enabled: bool
    semantic_cache_enabled: bool
    semantic_similarity_threshold: float  # 0–1; higher = stricter
    cache_max_size: int

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int          # candidates retrieved
    retrieval_rerank_top_k: int   # chunks passed to the model after reranking
    knowledge_base_path: str

    # ── Context ───────────────────────────────────────────────────────────────
    max_context_tokens: int   # rough token budget for context
    max_history_turns: int    # conversation turns kept verbatim

    # ── Routing thresholds ────────────────────────────────────────────────────
    complexity_low_threshold: float   # below → small model
    complexity_high_threshold: float  # above → large model
    confidence_escalation_threshold: float  # escalate when confidence < this


def _bool(val: str, default: bool) -> bool:
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def load_config() -> Config:
    """Read config from environment variables with safe defaults."""
    provider = os.getenv("LLM_PROVIDER", "mock")

    small = ModelConfig(
        name=os.getenv("SMALL_MODEL", "gpt-3.5-turbo"),
        tier="small",
        cost_per_1k_input=float(os.getenv("SMALL_COST_INPUT", "0.0005")),
        cost_per_1k_output=float(os.getenv("SMALL_COST_OUTPUT", "0.0015")),
        max_tokens=int(os.getenv("SMALL_MAX_TOKENS", "512")),
        provider=provider,
    )
    medium = ModelConfig(
        name=os.getenv("MEDIUM_MODEL", "gpt-4o-mini"),
        tier="medium",
        cost_per_1k_input=float(os.getenv("MEDIUM_COST_INPUT", "0.00015")),
        cost_per_1k_output=float(os.getenv("MEDIUM_COST_OUTPUT", "0.0006")),
        max_tokens=int(os.getenv("MEDIUM_MAX_TOKENS", "1024")),
        provider=provider,
    )
    large = ModelConfig(
        name=os.getenv("LARGE_MODEL", "gpt-4o"),
        tier="large",
        cost_per_1k_input=float(os.getenv("LARGE_COST_INPUT", "0.005")),
        cost_per_1k_output=float(os.getenv("LARGE_COST_OUTPUT", "0.015")),
        max_tokens=int(os.getenv("LARGE_MAX_TOKENS", "4096")),
        provider=provider,
    )

    return Config(
        llm_provider=provider,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        small_model=small,
        medium_model=medium,
        large_model=large,
        exact_cache_enabled=_bool(os.getenv("EXACT_CACHE_ENABLED"), True),
        semantic_cache_enabled=_bool(os.getenv("SEMANTIC_CACHE_ENABLED"), True),
        semantic_similarity_threshold=float(
            os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.90")
        ),
        cache_max_size=int(os.getenv("CACHE_MAX_SIZE", "500")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "10")),
        retrieval_rerank_top_k=int(os.getenv("RETRIEVAL_RERANK_TOP_K", "3")),
        knowledge_base_path=os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base.md"),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "3000")),
        max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "6")),
        complexity_low_threshold=float(os.getenv("COMPLEXITY_LOW_THRESHOLD", "0.35")),
        complexity_high_threshold=float(
            os.getenv("COMPLEXITY_HIGH_THRESHOLD", "0.65")
        ),
        confidence_escalation_threshold=float(
            os.getenv("CONFIDENCE_ESCALATION_THRESHOLD", "0.45")
        ),
    )
