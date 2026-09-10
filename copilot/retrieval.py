"""
Simple retrieval pipeline over a local knowledge base.

Pipeline
────────
1. Load & chunk a markdown knowledge base file into passages.
2. BM25-style TF-IDF scoring to retrieve top-k candidate chunks.
3. Cosine-similarity reranking to select the best rerank_top_k chunks.
4. Returns scored chunks with a composite retrieval quality score.

The design is intentionally dependency-free (no FAISS, no vector DB) so the
app runs anywhere.  Replace `_score_bm25` / `_rerank` with real embeddings
when you have a vector store.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from copilot.cache import _tokenize, _embed, _cosine


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str
    index: int
    score: float = 0.0


@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    quality_score: float   # 0–1; how confidently we retrieved relevant content
    raw_candidates: int    # how many chunks were initially retrieved


# ─── Knowledge base loading ───────────────────────────────────────────────────

_CHUNK_SEP = re.compile(r"\n#{1,3} ")   # split on markdown headings


def _split_into_chunks(text: str, source: str) -> List[Chunk]:
    """Split markdown text on headings into logical chunks."""
    parts = _CHUNK_SEP.split(text)
    chunks: List[Chunk] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if len(part) < 30:   # skip very short fragments
            continue
        chunks.append(Chunk(text=part, source=source, index=i))
    return chunks


def load_knowledge_base(path: str) -> List[Chunk]:
    """Load and chunk a markdown knowledge base file."""
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    return _split_into_chunks(raw, source=os.path.basename(path))


# ─── BM25-style scoring ───────────────────────────────────────────────────────

def _build_idf(chunks: List[Chunk]) -> dict:
    """Pre-compute per-token IDF across the corpus."""
    n = len(chunks)
    df: dict = {}
    for chunk in chunks:
        for tok in set(_tokenize(chunk.text)):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log((n + 1) / (freq + 1)) for tok, freq in df.items()}


def _score_bm25(
    query_tokens: List[str],
    chunk: Chunk,
    idf: dict,
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 100.0,
) -> float:
    """Approximate BM25 score for a chunk given query tokens."""
    tokens = _tokenize(chunk.text)
    dl = len(tokens)
    tf_map: dict = {}
    for tok in tokens:
        tf_map[tok] = tf_map.get(tok, 0) + 1

    score = 0.0
    for tok in query_tokens:
        if tok not in tf_map:
            continue
        tf = tf_map[tok]
        idf_val = idf.get(tok, 0.0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += idf_val * numerator / denominator
    return score


# ─── Retrieval pipeline ───────────────────────────────────────────────────────

class Retriever:
    def __init__(self, chunks: List[Chunk], top_k: int = 10, rerank_k: int = 3):
        self.chunks = chunks
        self.top_k = top_k
        self.rerank_k = rerank_k
        self._idf = _build_idf(chunks) if chunks else {}
        self._avg_dl = (
            sum(len(_tokenize(c.text)) for c in chunks) / len(chunks)
            if chunks else 100.0
        )

    def retrieve(self, query: str) -> RetrievalResult:
        """Run the full retrieval pipeline and return ranked chunks."""
        if not self.chunks:
            return RetrievalResult(chunks=[], quality_score=0.0, raw_candidates=0)

        query_tokens = _tokenize(query)
        if not query_tokens:
            return RetrievalResult(chunks=[], quality_score=0.0, raw_candidates=0)

        # Stage 1: BM25 scoring
        scored: List[Tuple[float, Chunk]] = []
        for chunk in self.chunks:
            s = _score_bm25(query_tokens, chunk, self._idf, avg_dl=self._avg_dl)
            if s > 0:
                scored.append((s, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = scored[: self.top_k]

        if not candidates:
            return RetrievalResult(chunks=[], quality_score=0.0, raw_candidates=0)

        # Stage 2: cosine re-rank
        q_emb = _embed(query)
        reranked: List[Tuple[float, Chunk]] = []
        for _, chunk in candidates:
            sim = _cosine(q_emb, _embed(chunk.text))
            reranked.append((sim, chunk))

        reranked.sort(key=lambda x: x[0], reverse=True)
        top = reranked[: self.rerank_k]

        final_chunks: List[Chunk] = []
        for sim, chunk in top:
            chunk.score = sim
            final_chunks.append(chunk)

        # Quality = mean similarity of top results (rough proxy for relevance)
        quality = sum(sim for sim, _ in top) / len(top) if top else 0.0

        return RetrievalResult(
            chunks=final_chunks,
            quality_score=float(quality),
            raw_candidates=len(candidates),
        )
