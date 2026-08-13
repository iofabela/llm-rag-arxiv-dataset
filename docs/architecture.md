# How it works — architecture

This page details the deep flow behind the chat app. For the three-minute
summary (problem, data, flow overview) see the [root README](../README.md).

## End-to-end flow

<p align="center">
  <img src="diagrams/architecture.svg" alt="ArXiv RAG chat architecture diagram" width="650">
</p>

## Components

1. **Streamlit UI** (`app_streamlit.py`) — chat input, renders the answer with
   an expandable list of cited sources, and 👍/👎 buttons per reply that call
   `POST /feedback`. It talks to the API through plain HTTP (`requests`),
   using `BACKEND_URL` from `.env` (default `http://localhost:8000`).

2. **FastAPI backend** (`src/arxiv_rag/api.py`) — exposes `GET /health`,
   `POST /chat`, `POST /feedback`, and an interactive Scalar client at
   `GET /scalar`. Full walkthrough in [api_scalar.md](api_scalar.md).

3. **RAG orchestration** (`src/arxiv_rag/rag.py`) — `answer_question()`
   orchestrates the whole request:
   - picks the retrieval strategy from `settings.retrieval_strategy`,
   - retrieves the top papers via that strategy,
   - formats them into a numbered context block,
   - sends `context + question` to the chat model,
   - returns the answer, cited sources, token usage, latency, and cost.

4. **Retrieval strategies** (`src/arxiv_rag/strategies.py`) — three pluggable
   strategies behind a single `(index, query, top_k) -> list[RankedResult]`
   interface, selected via `RETRIEVAL_STRATEGY`:

   | Strategy | Pipeline |
   |---|---|
   | `hybrid_rerank` (default, recommended) | NumPy/BM25 hybrid search + RRF, then a local cross-encoder rerank of the candidate pool |
   | `hybrid_only` | Same hybrid search + RRF, truncated to `top_k` — no rerank |
   | `elasticsearch_rrf` | Elasticsearch's native RRF retriever (BM25 + kNN), server side — requires Elasticsearch |

   - `hybrid_search` (`src/arxiv_rag/search.py`) fuses a dense ranking (dot
     product of the query embedding against the cached `embeddings.npy`) with
     a BM25 ranking, using Reciprocal Rank Fusion (`1 / (k + rank)`).
   - The reranker (`src/arxiv_rag/rerank.py`) is a local
     `cross-encoder/ms-marco-MiniLM-L-6-v2` — no extra API key or network call.

   Tuning knobs from `.env`:

   | Variable | Default | Meaning |
   |---|---|---|
   | `RETRIEVAL_STRATEGY` | `hybrid_rerank` | Strategy used by the API |
   | `HYBRID_CANDIDATE_K` | `100` | Candidate pool size for the NumPy/BM25 hybrid search |
   | `RERANK_TOP_K` | `5` | Papers kept for context after retrieval/rerank |
   | `RRF_K` | `60` | RRF constant |
   | `ES_URL` / `ES_API_KEY` | `http://localhost:9200` / empty | Elasticsearch endpoint (only for `elasticsearch_rrf`) |

5. **Generation & grounding** — the chat completion is grounded:
   the system prompt (`rag.SYSTEM_PROMPT`) forbids outside knowledge and
   invented facts and requires inline citations `[1]`, `[2]`, …; the user
   prompt is `Retrieved papers:\n\n{context}\n\nQuestion: {question}`. If no
   papers are retrieved, the API returns `NO_RESULTS_ANSWER` rather than
   fabricating an answer.

6. **Cost & token tracking** — `rag.MODEL_PRICING` maps each model to USD per
   1M tokens (input, output). Every `RagAnswer` carries `prompt_tokens`,
   `completion_tokens`, `total_tokens`, `response_time`, and `cost`, which the
   API logs to Postgres (see [monitoring_feedback.md](monitoring_feedback.md)).

7. **Indexing & data** — papers come from a Kaggle snapshot, loaded by dlt
   into DuckDB (`data/arxiv_papers.duckdb`) and then into an in-memory
   `RagIndex` (`src/arxiv_rag/indexing.py`): Pandas dataframe + cached OpenAI
   embeddings (`artifacts/embeddings.npy`) + BM25. Build/refresh with
   `scripts/build_index.py`. The optional Elasticsearch index is populated
   from the same embeddings (`es_search.index_to_elasticsearch`).

## Data sources & storage

| Piece | Where |
|---|---|
| Raw snapshot | `data/arxiv_ai.csv` (Kaggle) |
| DuckDB pipeline table | `data/arxiv_papers.duckdb` → `arxiv_data.arxiv_papers` |
| Embedding matrix | `artifacts/embeddings.npy` |
| Conversation/feedback logs | Postgres (`arxiv_rag`) |

## See also

- [api_scalar.md](api_scalar.md) — every HTTP endpoint, request/response examples
- [retrieval_evaluation.md](retrieval_evaluation.md) — how the strategies are compared
- [monitoring_feedback.md](monitoring_feedback.md) — feedback + Grafana
- [ingestion_pipeline.md](ingestion_pipeline.md) — dlt + Kestra