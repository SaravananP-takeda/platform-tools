"""
Intelligent routing – decides which model tier to use for each request.

Routing logic
─────────────
1. Score complexity from request signals (length, question structure, keywords).
2. Map the score to a tier: small / medium / large.
3. After inference, optionally escalate to the next tier if the model's
   confidence is too low or the answer failed basic validation.

Routing signals
───────────────
- prompt length (longer → more complex)
- multi-step / comparative / reasoning keywords
- number of question marks / sub-questions
- retrieval quality score (low quality → escalate)
- conversation depth
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from copilot.config import Config, ModelConfig

# ─── Complexity scoring ───────────────────────────────────────────────────────

_COMPLEXITY_KEYWORDS_HIGH = re.compile(
    r"\b(compare|contrast|analyze|synthesize|evaluate|explain why|"
    r"reasoning|relationship between|impact of|trade-off|strategy|"
    r"plan|design|architect|recommend|critique|pros and cons|"
    r"step by step|multi[-\s]step|walk me through|in depth|"
    r"comprehensive|detailed analysis|root cause|troubleshoot|debug)\b",
    re.IGNORECASE,
)

_COMPLEXITY_KEYWORDS_MEDIUM = re.compile(
    r"\b(summarize|describe|list|explain|how does|what is|"
    r"when should|best practice|overview|examples of|use case|"
    r"difference between|meaning of|define)\b",
    re.IGNORECASE,
)


def score_complexity(
    query: str,
    history_length: int = 0,
    retrieval_score: float = 1.0,
) -> float:
    """
    Return a complexity score in [0, 1].
    0 = trivial (good for small model)
    1 = highly complex (needs large model)
    """
    score = 0.0
    words = query.split()
    n_words = len(words)

    # Length component (up to 0.25)
    score += min(n_words / 120, 0.25)

    # High-complexity keyword bonus (up to 0.35)
    high_matches = len(_COMPLEXITY_KEYWORDS_HIGH.findall(query))
    score += min(high_matches * 0.12, 0.35)

    # Medium-complexity keywords (up to 0.15)
    medium_matches = len(_COMPLEXITY_KEYWORDS_MEDIUM.findall(query))
    score += min(medium_matches * 0.05, 0.15)

    # Multiple question marks or bullet-style sub-questions (up to 0.10)
    n_questions = query.count("?")
    score += min(n_questions * 0.05, 0.10)

    # Low retrieval quality → harder to answer from context → escalate (up to 0.15)
    score += max(0.0, (1.0 - retrieval_score) * 0.15)

    # Deep conversation means more context to manage (up to 0.10)
    score += min(history_length * 0.02, 0.10)

    return min(score, 1.0)


# ─── Routing decision ─────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    tier: str              # "small" | "medium" | "large"
    complexity_score: float
    model: ModelConfig
    reason: str


def route_request(
    query: str,
    cfg: Config,
    history_length: int = 0,
    retrieval_score: float = 1.0,
) -> RouteDecision:
    """Pick the cheapest model tier suitable for this request."""
    score = score_complexity(query, history_length, retrieval_score)

    if score < cfg.complexity_low_threshold:
        tier = "small"
        model = cfg.small_model
        reason = f"Low complexity ({score:.2f} < {cfg.complexity_low_threshold})"
    elif score > cfg.complexity_high_threshold:
        tier = "large"
        model = cfg.large_model
        reason = f"High complexity ({score:.2f} > {cfg.complexity_high_threshold})"
    else:
        tier = "medium"
        model = cfg.medium_model
        reason = f"Medium complexity ({score:.2f})"

    return RouteDecision(
        tier=tier,
        complexity_score=score,
        model=model,
        reason=reason,
    )


# ─── Escalation ───────────────────────────────────────────────────────────────

def should_escalate(
    confidence: float,
    current_tier: str,
    cfg: Config,
) -> Tuple[bool, Optional[ModelConfig]]:
    """
    Return (True, next_model) when the current answer confidence is too low
    and a stronger model is available.
    """
    if confidence >= cfg.confidence_escalation_threshold:
        return False, None

    if current_tier == "small":
        return True, cfg.medium_model
    if current_tier == "medium":
        return True, cfg.large_model
    # Already on large; no further escalation
    return False, None


def get_tier_chain(cfg: Config) -> List[ModelConfig]:
    """Return models in ascending tier order."""
    return [cfg.small_model, cfg.medium_model, cfg.large_model]
