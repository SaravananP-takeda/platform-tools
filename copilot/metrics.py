"""
Metrics store – tracks all KPIs for the Enterprise Copilot.

Tracked metrics
───────────────
Per-request:
  - cost (input + output tokens × per-token pricing)
  - input_tokens / output_tokens
  - latency_ms
  - tier used (small / medium / large)
  - cache_type (none / exact / semantic)
  - n_chunks_retrieved
  - answer_quality (0–1 float, heuristic or user rating)
  - escalated (bool)
  - failed (bool)

Derived / aggregated:
  - cost_per_request
  - cost_per_successful_task
  - cache_hit_rate
  - semantic_cache_reuse_rate
  - avg_chunks_per_request
  - escalation_rate
  - failure_rate
  - quality_adjusted_cost_per_successful_outcome
  - latency_by_route
  - quality_by_route
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional


# ─── Per-request record ───────────────────────────────────────────────────────

@dataclass
class RequestRecord:
    query: str
    tier: str                     # "small" | "medium" | "large" | "cache"
    cache_type: str               # "none" | "exact" | "semantic"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    n_chunks: int
    confidence: float
    escalated: bool
    failed: bool
    answer_quality: float         # 0–1; estimated or user-provided
    timestamp: float = field(default_factory=time.time)


# ─── Metrics store ───────────────────────────────────────────────────────────

class MetricsStore:
    """In-memory store for all copilot metrics."""

    def __init__(self) -> None:
        self.records: List[RequestRecord] = []

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, rec: RequestRecord) -> None:
        self.records.append(rec)

    # ── Computed metrics ─────────────────────────────────────────────────────

    @property
    def total_requests(self) -> int:
        return len(self.records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def cost_per_request(self) -> float:
        if not self.records:
            return 0.0
        return self.total_cost_usd / len(self.records)

    @property
    def cost_per_successful_task(self) -> float:
        successful = [r for r in self.records if not r.failed]
        if not successful:
            return 0.0
        return sum(r.cost_usd for r in successful) / len(successful)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def cache_hit_rate(self) -> float:
        if not self.records:
            return 0.0
        hits = sum(1 for r in self.records if r.cache_type == "exact")
        return hits / len(self.records)

    @property
    def semantic_cache_reuse_rate(self) -> float:
        if not self.records:
            return 0.0
        hits = sum(1 for r in self.records if r.cache_type == "semantic")
        return hits / len(self.records)

    @property
    def avg_chunks_per_request(self) -> float:
        live = [r for r in self.records if r.cache_type == "none"]
        if not live:
            return 0.0
        return mean(r.n_chunks for r in live)

    @property
    def escalation_rate(self) -> float:
        live = [r for r in self.records if r.cache_type == "none"]
        if not live:
            return 0.0
        return sum(1 for r in live if r.escalated) / len(live)

    @property
    def failure_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.failed) / len(self.records)

    @property
    def avg_answer_quality(self) -> float:
        live = [r for r in self.records if not r.failed]
        if not live:
            return 0.0
        return mean(r.answer_quality for r in live)

    @property
    def quality_adjusted_cost_per_successful_outcome(self) -> float:
        """
        Penalises cost for low quality answers.
        quality-adjusted cost = actual cost / quality_score
        so a half-quality answer costs twice as much per "good outcome".
        """
        successful = [r for r in self.records if not r.failed and r.answer_quality > 0]
        if not successful:
            return 0.0
        return mean(
            r.cost_usd / r.answer_quality for r in successful
        )

    @property
    def latency_by_route(self) -> Dict[str, float]:
        """Mean latency (ms) grouped by tier."""
        tiers: Dict[str, List[float]] = {}
        for r in self.records:
            tiers.setdefault(r.tier, []).append(r.latency_ms)
        return {t: mean(v) for t, v in tiers.items()}

    @property
    def quality_by_route(self) -> Dict[str, float]:
        """Mean answer quality grouped by tier."""
        tiers: Dict[str, List[float]] = {}
        for r in self.records:
            tiers.setdefault(r.tier, []).append(r.answer_quality)
        return {t: mean(v) for t, v in tiers.items()}

    @property
    def cost_by_route(self) -> Dict[str, float]:
        """Mean cost per request grouped by tier."""
        tiers: Dict[str, List[float]] = {}
        for r in self.records:
            tiers.setdefault(r.tier, []).append(r.cost_usd)
        return {t: mean(v) for t, v in tiers.items()}

    @property
    def requests_by_route(self) -> Dict[str, int]:
        tiers: Dict[str, int] = {}
        for r in self.records:
            tiers[r.tier] = tiers.get(r.tier, 0) + 1
        return tiers

    # ── Snapshot for dashboard ────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return all metrics as a plain dict for display."""
        return {
            "total_requests": self.total_requests,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_per_request": round(self.cost_per_request, 6),
            "cost_per_successful_task": round(self.cost_per_successful_task, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "semantic_cache_reuse_rate": round(self.semantic_cache_reuse_rate, 4),
            "avg_chunks_per_request": round(self.avg_chunks_per_request, 2),
            "escalation_rate": round(self.escalation_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "avg_answer_quality": round(self.avg_answer_quality, 4),
            "quality_adjusted_cost": round(
                self.quality_adjusted_cost_per_successful_outcome, 6
            ),
            "latency_by_route": {
                k: round(v, 1) for k, v in self.latency_by_route.items()
            },
            "quality_by_route": {
                k: round(v, 3) for k, v in self.quality_by_route.items()
            },
            "cost_by_route": {
                k: round(v, 6) for k, v in self.cost_by_route.items()
            },
            "requests_by_route": self.requests_by_route,
        }


# ─── Cost calculation helper ─────────────────────────────────────────────────

def compute_cost(
    input_tokens: int,
    output_tokens: int,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
) -> float:
    """Return total USD cost for a request."""
    return (input_tokens / 1000) * cost_per_1k_input + (
        output_tokens / 1000
    ) * cost_per_1k_output


# ─── Answer quality heuristic ─────────────────────────────────────────────────

def estimate_quality(response: str, confidence: float) -> float:
    """
    Heuristic quality score combining:
    - model confidence (primary signal)
    - response length (very short → likely low quality)
    - absence of failure indicators
    """
    if not response or len(response.strip()) < 20:
        return 0.1

    failure_phrases = [
        "i don't know", "i cannot", "i'm unable", "no information",
        "not available", "cannot answer", "out of scope",
    ]
    lower = response.lower()
    if any(p in lower for p in failure_phrases):
        return max(0.1, confidence * 0.5)

    length_score = min(len(response) / 400, 1.0)  # plateaus at ~400 chars
    return round(confidence * 0.7 + length_score * 0.3, 3)
