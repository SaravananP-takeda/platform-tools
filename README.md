# platform-tools – Enterprise Copilot

A production-ready **Enterprise AI Copilot** with cost-optimisation techniques,
built on Streamlit.  The system combines intelligent model routing, exact and
semantic caching, RAG retrieval, context compression, and a full metrics
dashboard — all runnable out of the box without any API keys.

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/SaravananP-takeda/platform-tools.git
cd platform-tools

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) copy and edit environment variables
cp .env.example .env
# edit .env to set LLM_PROVIDER=openai and OPENAI_API_KEY if you want real calls

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501** with two tabs:

| Tab | Purpose |
|-----|---------|
| 💬 **Chat** | Conversational copilot interface |
| 📊 **Metrics** | Live KPI dashboard |

---

## Architecture

```
app.py                    ← Streamlit entry point
copilot/
  config.py               ← Environment-driven configuration
  providers.py            ← LLM abstraction (mock + OpenAI)
  routing.py              ← Tiered model routing & escalation
  cache.py                ← Exact-match + semantic cache
  retrieval.py            ← BM25 + cosine reranking RAG pipeline
  context.py              ← History trimming, dedup, compression
  metrics.py              ← Full KPI store & cost calculations
data/
  knowledge_base.md       ← Enterprise knowledge content (editable)
requirements.txt
.env.example
```

---

## Cost-optimisation techniques

### 1 – Intelligent routing

Every request is scored for complexity using:
- prompt length
- reasoning/multi-step keywords
- conversation depth
- retrieval quality

The score maps to a model tier:

| Tier | Model | When used |
|------|-------|-----------|
| Small | `gpt-3.5-turbo` | Simple lookups, FAQs, classifications |
| Medium | `gpt-4o-mini` | Summarisation, standard RAG answers |
| Large | `gpt-4o` | Complex reasoning, multi-document analysis |

**Escalation**: if the chosen model returns a confidence score below the
threshold, the request is automatically re-sent to the next tier.

### 2 – Caching

Two layers of caching, both configurable:

- **Exact-match cache** – SHA-256 hash of the query; zero cost on repeat hits.
- **Semantic cache** – TF-IDF vector cosine similarity; reuses answers for
  semantically equivalent questions (default threshold: 90 %).

Both layers use an LRU eviction policy.

### 3 – Context optimisation

Before sending to the LLM:
- Retrieved chunks are **deduplicated** (Jaccard overlap > 50 % removed).
- Chunks are **compressed** to fit a token budget (default 3,000 tokens).
- Conversation history is **trimmed** to the last N turns (default 6).
- Budget is split 60 / 40 between retrieved context and history.

### 4 – Retrieval pipeline

1. **BM25 scoring** over the knowledge base (no external vector DB needed).
2. **Cosine reranking** on TF-IDF embeddings to select top-k chunks.
3. Retrieval quality score influences routing (low quality → escalate tier).

---

## Metrics tracked

| Metric | Description |
|--------|-------------|
| Cost per request | Average USD cost across all requests |
| Cost per successful task | Cost excluding failed requests |
| Input / output tokens | Running totals |
| Cache hit rate | Fraction of requests served from exact cache |
| Semantic cache reuse rate | Fraction served from semantic cache |
| Avg chunks per request | Mean retrieved passages used |
| Escalation rate | Fraction of requests that escalated to a larger model |
| Latency by route | Mean response time per tier (ms) |
| Answer quality by route | Mean quality score per tier |
| Failure / fallback rate | Fraction of requests that failed |
| **Quality-adjusted cost** | `cost / quality_score` – the primary derived KPI |

---

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
# Use mock provider (no API key needed — default)
LLM_PROVIDER=mock

# Switch to real OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Tune routing thresholds
COMPLEXITY_LOW_THRESHOLD=0.35
COMPLEXITY_HIGH_THRESHOLD=0.65
CONFIDENCE_ESCALATION_THRESHOLD=0.45

# Cache settings
SEMANTIC_SIMILARITY_THRESHOLD=0.90
CACHE_MAX_SIZE=500

# Retrieval settings
RETRIEVAL_TOP_K=10
RETRIEVAL_RERANK_TOP_K=3
KNOWLEDGE_BASE_PATH=data/knowledge_base.md
```

---

## Adding your own knowledge base

Edit `data/knowledge_base.md` (or point `KNOWLEDGE_BASE_PATH` to another file).
The retriever splits on markdown headings (`#`, `##`, `###`) into chunks.
No re-indexing step needed — chunks are loaded at startup.

---

## Swapping LLM providers

To add a new provider (e.g. Anthropic, Azure, Bedrock):

1. Add a new function `_anthropic_respond(...)` in `copilot/providers.py`
   following the same signature as `_openai_respond`.
2. Add the new branch in `LLMProvider.complete()`.
3. Set `LLM_PROVIDER=anthropic` in `.env`.

---

## Project structure rationale

| Module | Responsibility |
|--------|---------------|
| `config.py` | Single source of truth for all settings |
| `providers.py` | Isolates LLM API details from app logic |
| `routing.py` | Pure routing logic, easy to unit-test |
| `cache.py` | Self-contained cache with no external deps |
| `retrieval.py` | RAG pipeline; replace internals with FAISS/Pinecone later |
| `context.py` | Context assembly strategy, independent of model choice |
| `metrics.py` | All KPIs in one place; easy to export to Prometheus/Grafana |
| `app.py` | Thin Streamlit shell; delegates all logic to modules |

---

## Code quality (SonarCloud)

This repository includes a GitHub Actions workflow
(`.github/workflows/sonarcloud.yml`) and a `sonar-project.properties` file
that scan the codebase with [SonarCloud](https://sonarcloud.io) on every push
to `copilot/build-enterprise-copilot-streamlit` and on every pull request.

To finish enabling SonarCloud analysis for this repository:

1. Import/create the project in SonarCloud under your organization.
2. Update the `sonar.organization` value in `sonar-project.properties` with
   your actual SonarCloud organization key (the placeholder value may not
   match your organization).
3. Add a repository secret named `SONAR_TOKEN` containing a SonarCloud
   analysis token (Settings → Secrets and variables → Actions).
4. Confirm the project key `SaravananP-takeda_platform-tools` matches the
   project you created in SonarCloud (or update it in
   `sonar-project.properties`).
