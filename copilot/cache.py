"""
Caching layer – two complementary strategies:

1. Exact-match cache
   Key = SHA-256 of the raw query string (case-folded, stripped).
   Zero-latency lookup for literally repeated questions.

2. Semantic cache
   Key = cosine similarity between the new query's embedding and stored embeddings.
   If similarity ≥ threshold, reuse the cached response.
   Uses a lightweight word-frequency (TF-IDF-style) embedding that requires no
   external ML libraries.  Swap `_embed()` for sentence-transformers or the
   OpenAI embeddings API when moving to production.

Cache eviction: simple LRU (Least Recently Used) up to `max_size` entries.
"""

from __future__ import annotations

import collections
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    query: str
    response: str
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


# ─── Lightweight text embedding ───────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can at in on of to for "
    "and or but not this that these those i you he she we they it".split()
)


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _embed(text: str) -> Dict[str, float]:
    """
    Build a normalised TF vector from text tokens.
    Suitable for cosine similarity; no external dependencies required.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}
    freq: Dict[str, float] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in freq.values()))
    if norm == 0:
        return freq
    return {k: v / norm for k, v in freq.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF vectors."""
    dot = sum(a.get(tok, 0.0) * val for tok, val in b.items())
    return dot  # already normalised


# ─── Cache store ─────────────────────────────────────────────────────────────

class ResponseCache:
    """Thread-safe (GIL) in-memory cache with exact and semantic lookup."""

    def __init__(self, max_size: int = 500, semantic_threshold: float = 0.90):
        self.max_size = max_size
        self.semantic_threshold = semantic_threshold

        # Exact cache: hash → CacheEntry (ordered for LRU)
        self._exact: "collections.OrderedDict[str, CacheEntry]" = (
            collections.OrderedDict()
        )

        # Semantic cache: list of (embedding, CacheEntry)
        self._semantic: List[Tuple[Dict[str, float], CacheEntry]] = []

        # Stats
        self.exact_hits: int = 0
        self.semantic_hits: int = 0
        self.total_lookups: int = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def _evict_if_needed(self) -> None:
        while len(self._exact) >= self.max_size:
            self._exact.popitem(last=False)  # evict oldest
        # Keep semantic list in sync
        while len(self._semantic) >= self.max_size:
            self._semantic.pop(0)

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup_exact(self, query: str) -> Optional[str]:
        """Return cached response for an exact query match, or None."""
        self.total_lookups += 1
        key = self._key(query)
        entry = self._exact.get(key)
        if entry is not None:
            entry.hit_count += 1
            self._exact.move_to_end(key)  # refresh LRU order
            self.exact_hits += 1
            return entry.response
        return None

    def lookup_semantic(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Return (cached_response, similarity_score) if a semantically similar
        query exists above the threshold, otherwise None.
        """
        if not self._semantic:
            return None

        q_emb = _embed(query)
        if not q_emb:
            return None

        best_score = 0.0
        best_entry: Optional[CacheEntry] = None
        for emb, entry in self._semantic:
            score = _cosine(q_emb, emb)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.semantic_threshold and best_entry is not None:
            best_entry.hit_count += 1
            self.semantic_hits += 1
            return best_entry.response, best_score
        return None

    def store(self, query: str, response: str) -> None:
        """Save a query-response pair to both exact and semantic caches."""
        self._evict_if_needed()

        key = self._key(query)
        entry = CacheEntry(query=query, response=response)

        self._exact[key] = entry
        self._exact.move_to_end(key)

        emb = _embed(query)
        if emb:
            self._semantic.append((emb, entry))

    def hit_rate(self) -> float:
        """Exact cache hit rate."""
        if self.total_lookups == 0:
            return 0.0
        return self.exact_hits / self.total_lookups

    def semantic_reuse_rate(self) -> float:
        """Semantic cache reuse rate (fraction of all lookups)."""
        if self.total_lookups == 0:
            return 0.0
        return self.semantic_hits / self.total_lookups

    def size(self) -> int:
        return len(self._exact)

    def clear(self) -> None:
        self._exact.clear()
        self._semantic.clear()
        self.exact_hits = 0
        self.semantic_hits = 0
        self.total_lookups = 0
