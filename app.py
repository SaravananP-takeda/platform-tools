"""
Enterprise Copilot – Streamlit Application
==========================================

Run:
    streamlit run app.py

The app has two tabs:
  1. 💬 Chat   – conversational copilot with cost-optimised inference
  2. 📊 Metrics – live dashboard of all KPIs

All heavy lifting lives in the `copilot/` package modules.
"""

from __future__ import annotations

import os
import sys
import time

import streamlit as st

# Make the package importable when running from the repo root
sys.path.insert(0, os.path.dirname(__file__))

# Load .env file if present (development convenience)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

from copilot.cache import ResponseCache
from copilot.config import load_config
from copilot.context import build_context
from copilot.metrics import (
    MetricsStore,
    RequestRecord,
    compute_cost,
    estimate_quality,
)
from copilot.providers import LLMProvider, Message
from copilot.retrieval import Retriever, load_knowledge_base
from copilot.routing import route_request, should_escalate

# ─── Page config (must be first Streamlit call) ───────────────────────────────

st.set_page_config(
    page_title="Enterprise Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session-state initialisation ─────────────────────────────────────────────

@st.cache_resource
def get_shared_resources():
    """
    Load config, provider, retriever, cache, and metrics store once.
    Cached across Streamlit reruns so state persists in the session.
    """
    cfg = load_config()
    provider = LLMProvider(
        llm_provider=cfg.llm_provider,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )
    chunks = load_knowledge_base(cfg.knowledge_base_path)
    retriever = Retriever(chunks, top_k=cfg.retrieval_top_k, rerank_k=cfg.retrieval_rerank_top_k)
    cache = ResponseCache(
        max_size=cfg.cache_max_size,
        semantic_threshold=cfg.semantic_similarity_threshold,
    )
    metrics = MetricsStore()
    return cfg, provider, retriever, cache, metrics


def _init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # List[dict] with role/content keys
    if "last_meta" not in st.session_state:
        st.session_state.last_meta = {}


# ─── Inference pipeline ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Enterprise Copilot for a technology company.
You are helpful, concise, and accurate. You answer questions about engineering
practices, platform tools, HR policies, IT support, AI/ML systems, and general
company knowledge.
If you are unsure, say so clearly rather than guessing.
Always prefer information from the provided context over prior knowledge."""


def run_inference(
    query: str,
    history: list[Message],
    cfg,
    provider: LLMProvider,
    retriever: Retriever,
    cache: ResponseCache,
    metrics: MetricsStore,
) -> tuple[str, dict]:
    """
    Full inference pipeline:
    1. Exact cache lookup
    2. Semantic cache lookup
    3. Retrieval
    4. Routing decision
    5. LLM call (with possible escalation)
    6. Store in cache + metrics
    """
    t_start = time.perf_counter()
    meta: dict = {}

    # ── 1. Exact cache ────────────────────────────────────────────────────────
    if cfg.exact_cache_enabled:
        cached = cache.lookup_exact(query)
        if cached is not None:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            meta = {
                "tier": "cache",
                "cache_type": "exact",
                "latency_ms": elapsed_ms,
                "n_chunks": 0,
                "cost_usd": 0.0,
                "escalated": False,
                "confidence": 1.0,
            }
            metrics.record(RequestRecord(
                query=query,
                tier="cache",
                cache_type="exact",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=elapsed_ms,
                n_chunks=0,
                confidence=1.0,
                escalated=False,
                failed=False,
                answer_quality=0.95,
            ))
            return cached, meta

    # ── 2. Semantic cache ─────────────────────────────────────────────────────
    if cfg.semantic_cache_enabled:
        sem_result = cache.lookup_semantic(query)
        if sem_result is not None:
            sem_response, similarity = sem_result
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            meta = {
                "tier": "cache",
                "cache_type": "semantic",
                "similarity": round(similarity, 3),
                "latency_ms": elapsed_ms,
                "n_chunks": 0,
                "cost_usd": 0.0,
                "escalated": False,
                "confidence": similarity,
            }
            metrics.record(RequestRecord(
                query=query,
                tier="cache",
                cache_type="semantic",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=elapsed_ms,
                n_chunks=0,
                confidence=similarity,
                escalated=False,
                failed=False,
                answer_quality=min(0.90, similarity),
            ))
            return sem_response, meta

    # ── 3. Retrieval ──────────────────────────────────────────────────────────
    retrieval_result = retriever.retrieve(query)
    chunks = retrieval_result.chunks
    n_chunks = len(chunks)

    # ── 4. Routing ────────────────────────────────────────────────────────────
    decision = route_request(
        query=query,
        cfg=cfg,
        history_length=len(history),
        retrieval_score=retrieval_result.quality_score,
    )
    model = decision.model
    escalated = False

    # ── 5. Context assembly ───────────────────────────────────────────────────
    messages, context_texts = build_context(
        query=query,
        history=history,
        chunks=chunks,
        system_prompt=SYSTEM_PROMPT,
        max_context_tokens=cfg.max_context_tokens,
        max_history_turns=cfg.max_history_turns,
    )

    # ── 6. LLM call ───────────────────────────────────────────────────────────
    try:
        response = provider.complete(messages, model, context_chunks=context_texts)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        error_text = f"⚠️ Error calling model: {exc}"
        metrics.record(RequestRecord(
            query=query,
            tier=decision.tier,
            cache_type="none",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=elapsed_ms,
            n_chunks=n_chunks,
            confidence=0.0,
            escalated=False,
            failed=True,
            answer_quality=0.0,
        ))
        return error_text, {"tier": decision.tier, "failed": True, "error": str(exc)}

    # ── 7. Escalation check ───────────────────────────────────────────────────
    do_escalate, escalation_model = should_escalate(response.confidence, decision.tier, cfg)
    if do_escalate and escalation_model is not None:
        escalated = True
        try:
            escalation_response = provider.complete(
                messages, escalation_model, context_chunks=context_texts
            )
            # Use escalated response if it's better (higher confidence)
            if escalation_response.confidence > response.confidence:
                model = escalation_model
                response = escalation_response
        except Exception:
            pass  # Keep the original response on escalation failure

    # ── 8. Cost calculation ───────────────────────────────────────────────────
    cost = compute_cost(
        response.input_tokens,
        response.output_tokens,
        model.cost_per_1k_input,
        model.cost_per_1k_output,
    )
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    quality = estimate_quality(response.content, response.confidence)

    # ── 9. Cache store ────────────────────────────────────────────────────────
    cache.store(query, response.content)

    # ── 10. Metrics record ────────────────────────────────────────────────────
    metrics.record(RequestRecord(
        query=query,
        tier=model.tier,
        cache_type="none",
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=cost,
        latency_ms=elapsed_ms,
        n_chunks=n_chunks,
        confidence=response.confidence,
        escalated=escalated,
        failed=False,
        answer_quality=quality,
    ))

    meta = {
        "tier": model.tier,
        "model": response.model_name,
        "cache_type": "none",
        "route_reason": decision.reason,
        "complexity_score": round(decision.complexity_score, 3),
        "escalated": escalated,
        "latency_ms": round(elapsed_ms, 1),
        "n_chunks": n_chunks,
        "retrieval_quality": round(retrieval_result.quality_score, 3),
        "cost_usd": round(cost, 6),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "confidence": round(response.confidence, 3),
        "answer_quality": round(quality, 3),
    }
    return response.content, meta


# ─── Chat tab ─────────────────────────────────────────────────────────────────

def render_chat(cfg, provider, retriever, cache, metrics):
    st.title("🤖 Enterprise Copilot")
    st.caption(
        f"Provider: **{cfg.llm_provider}** · "
        f"Routing: small / medium / large · "
        f"Cache: exact + semantic · "
        f"KB chunks loaded: **{len(retriever.chunks)}**"
    )

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                _render_message_meta(msg["meta"])

    # Chat input
    if prompt := st.chat_input("Ask anything about our platform, tools, or policies…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build history as Message objects (exclude system, handled by build_context)
        history = [
            Message(role=m["role"], content=m["content"])
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, meta = run_inference(
                    query=prompt,
                    history=history,
                    cfg=cfg,
                    provider=provider,
                    retriever=retriever,
                    cache=cache,
                    metrics=metrics,
                )
            st.markdown(answer)
            _render_message_meta(meta)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "meta": meta}
        )


def _render_message_meta(meta: dict):
    """Render small expandable metadata pill under each assistant message."""
    if not meta:
        return
    cache_type = meta.get("cache_type", "none")
    tier = meta.get("tier", "?")

    label_parts = []
    if cache_type == "exact":
        label_parts.append("✅ Exact cache hit")
    elif cache_type == "semantic":
        label_parts.append(f"🔄 Semantic cache ({meta.get('similarity', '')})")
    else:
        label_parts.append(f"🔀 Route: **{tier}**")
        if meta.get("escalated"):
            label_parts.append("⬆️ Escalated")

    label_parts.append(f"⏱ {meta.get('latency_ms', 0):.0f} ms")
    if meta.get("cost_usd", 0) > 0:
        label_parts.append(f"💰 ${meta['cost_usd']:.5f}")

    with st.expander(" · ".join(label_parts), expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if cache_type == "none":
                st.write(f"**Model:** {meta.get('model', tier)}")
                st.write(f"**Complexity:** {meta.get('complexity_score', '-')}")
                st.write(f"**Route reason:** {meta.get('route_reason', '-')}")
                st.write(f"**Retrieval quality:** {meta.get('retrieval_quality', '-')}")
                st.write(f"**Retrieved chunks:** {meta.get('n_chunks', 0)}")
        with col2:
            if cache_type == "none":
                st.write(f"**Input tokens:** {meta.get('input_tokens', 0)}")
                st.write(f"**Output tokens:** {meta.get('output_tokens', 0)}")
                st.write(f"**Confidence:** {meta.get('confidence', '-')}")
                st.write(f"**Answer quality:** {meta.get('answer_quality', '-')}")


# ─── Metrics tab ──────────────────────────────────────────────────────────────

def render_metrics(metrics: MetricsStore, cache: ResponseCache):
    st.title("📊 Copilot Metrics Dashboard")

    if metrics.total_requests == 0:
        st.info("No requests yet. Start chatting to see metrics.")
        return

    snap = metrics.snapshot()

    # ── Top-level KPIs ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", snap["total_requests"])
    col2.metric("Total Cost (USD)", f"${snap['total_cost_usd']:.5f}")
    col3.metric("Cost / Request", f"${snap['cost_per_request']:.5f}")
    col4.metric("Cost / Successful Task", f"${snap['cost_per_successful_task']:.5f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Exact Cache Hit Rate", f"{snap['cache_hit_rate']:.1%}")
    col6.metric("Semantic Cache Reuse", f"{snap['semantic_cache_reuse_rate']:.1%}")
    col7.metric("Escalation Rate", f"{snap['escalation_rate']:.1%}")
    col8.metric("Failure Rate", f"{snap['failure_rate']:.1%}")

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Total Input Tokens", snap["total_input_tokens"])
    col10.metric("Total Output Tokens", snap["total_output_tokens"])
    col11.metric("Avg Chunks / Request", snap["avg_chunks_per_request"])
    col12.metric("Avg Answer Quality", f"{snap['avg_answer_quality']:.2%}")

    st.metric(
        "⭐ Quality-Adjusted Cost / Successful Outcome",
        f"${snap['quality_adjusted_cost']:.5f}",
        help=(
            "Actual cost divided by quality score. "
            "Lower is better. Penalises cheap-but-poor answers."
        ),
    )

    st.divider()

    # ── Route breakdown ───────────────────────────────────────────────────────
    st.subheader("By Route")
    import pandas as pd  # imported here to avoid startup delay

    routes = snap["requests_by_route"]
    latency = snap["latency_by_route"]
    quality = snap["quality_by_route"]
    cost_r = snap["cost_by_route"]

    all_tiers = sorted(set(list(routes) + list(latency) + list(quality) + list(cost_r)))
    rows = []
    for t in all_tiers:
        rows.append({
            "Route": t,
            "Requests": routes.get(t, 0),
            "Avg Latency (ms)": latency.get(t, "-"),
            "Avg Quality": quality.get(t, "-"),
            "Avg Cost (USD)": cost_r.get(t, "-"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Route"), use_container_width=True)

    st.divider()

    # ── Cache stats ───────────────────────────────────────────────────────────
    st.subheader("Cache")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cache Size", cache.size())
    c2.metric("Exact Hits", cache.exact_hits)
    c3.metric("Semantic Hits", cache.semantic_hits)

    st.divider()

    # ── Request log ───────────────────────────────────────────────────────────
    st.subheader("Recent Requests")
    recent = metrics.records[-20:][::-1]
    log_rows = []
    for r in recent:
        log_rows.append({
            "Query (truncated)": r.query[:60] + ("…" if len(r.query) > 60 else ""),
            "Tier": r.tier,
            "Cache": r.cache_type,
            "Tokens In": r.input_tokens,
            "Tokens Out": r.output_tokens,
            "Cost ($)": f"{r.cost_usd:.5f}",
            "Latency (ms)": f"{r.latency_ms:.0f}",
            "Quality": f"{r.answer_quality:.2f}",
            "Escalated": "✓" if r.escalated else "",
            "Failed": "✗" if r.failed else "",
        })
    if log_rows:
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(cfg, cache: ResponseCache):
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.write(f"**Provider:** `{cfg.llm_provider}`")
        st.write(f"**Small model:** `{cfg.small_model.name}`")
        st.write(f"**Medium model:** `{cfg.medium_model.name}`")
        st.write(f"**Large model:** `{cfg.large_model.name}`")
        st.write(f"**Complexity thresholds:** {cfg.complexity_low_threshold} / {cfg.complexity_high_threshold}")
        st.write(f"**Semantic threshold:** {cfg.semantic_similarity_threshold}")
        st.write(f"**Retrieval top-k:** {cfg.retrieval_top_k} → rerank {cfg.retrieval_rerank_top_k}")

        st.divider()
        st.header("🗂️ Cache")
        st.write(f"**Exact cache:** {'✅ on' if cfg.exact_cache_enabled else '❌ off'}")
        st.write(f"**Semantic cache:** {'✅ on' if cfg.semantic_cache_enabled else '❌ off'}")
        st.write(f"**Cache size:** {cache.size()} / {cfg.cache_max_size}")

        if st.button("🗑️ Clear cache"):
            cache.clear()
            st.success("Cache cleared.")

        st.divider()
        if st.button("🔄 Reset conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(
            "Enterprise Copilot · cost-optimised inference with "
            "intelligent routing, caching, and RAG"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    _init_session()
    cfg, provider, retriever, cache, metrics = get_shared_resources()

    render_sidebar(cfg, cache)

    tab_chat, tab_metrics = st.tabs(["💬 Chat", "📊 Metrics"])

    with tab_chat:
        render_chat(cfg, provider, retriever, cache, metrics)

    with tab_metrics:
        render_metrics(metrics, cache)


if __name__ == "__main__":
    main()
