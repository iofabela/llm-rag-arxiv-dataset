# ArXiv AI Papers RAG Chat

Ask natural-language questions about **10,000 AI/ML research papers** from arXiv
and get an answer **grounded in the retrieved papers, with citations**, plus
👍/👎 feedback and a live Grafana monitoring dashboard.

Capstone project for the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

## Why this project

arXiv publishes thousands of AI/ML papers, but its search only returns *lists
of papers*, not answers. It won't *answer* a question. To find out *"what methods do recent
papers use to evaluate RAG systems?"* you'd have to skim dozens of abstracts
and synthesize the answer yourself. This project is a **RAG
(Retrieval-Augmented Generation)** app that does that for you:

1. Ask a question in plain English in a web chat UI.
2. The app retrieves the most relevant papers with **hybrid search + reranking**.
3. An LLM answers using **only** those papers and cites them inline. If the
   papers don't contain the answer, it says so instead of guessing.
4. Every conversation is logged with cost/latency/tokens, you can rate each
   answer 👍/👎, and a Grafana dashboard monitors the whole system.

## Table of contents

- [Problem description](#problem-description)
- [Data](#data)
- [How it works (summary)](#how-it-works-summary)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Documentation](#documentation)
- [Tech stack](#tech-stack)

## Problem description

The project solves the "list of papers vs. answer" problem described in
[Why this project](#why-this-project): a RAG pipeline that retrieves relevant
arXiv papers and produces a **grounded, cited answer**, plus the infrastructure
(ingestion, monitoring, evaluation) to keep it reliable.

## Data

- **Source**: Kaggle dataset
  [`yasirabdaali/arxivorg-ai-research-papers-dataset`](https://www.kaggle.com/datasets/yasirabdaali/arxivorg-ai-research-papers-dataset), packaged as
  10,000 AI research papers from arXiv.
- **Fields used**: `title`, `authors`, `summary` (abstract), `published`,
  `entry_id` (arXiv id), `pdf_url`.
- **Storage**: downloaded to `data/arxiv_ai.csv`, loaded by a dlt pipeline into
  DuckDB (`data/arxiv_papers.duckdb`), then into an in-memory index of
  embeddings + BM25. The app reads DuckDB when present and falls back to CSV.
- **Refresh**: a Kestra flow re-ingests the dataset daily
  (see [Ingestion pipeline](docs/ingestion_pipeline.md)).

## How it works (summary)

<p align="center">
  <img src="docs/diagrams/architecture.svg" alt="ArXiv RAG chat architecture diagram" width="800">
</p>

1. **Retrieve**. The question is matched against every paper via the
   configured strategy. The recommended default is `hybrid_rerank`: RRF fusion
   of BM25 + dense embeddings, then a local cross-encoder rerank.
2. **Generate**. The top papers become a numbered context block for
   `gpt-4o-mini`, which answers strictly grounded in that context and cites
   papers as `[1]`, `[2]`, …
3. **Log**. Every `/chat` call (question, answer, strategy, tokens, cost,
   latency) is written to Postgres.
4. **Feedback**. 👍/👎 on each answer, stored via `POST /feedback`.
5. **Monitor**. Grafana visualizes volume, latency, cost, tokens, and feedback.

The full deep dive (strategies, prompts, cost tracking, indexing) is in
**[How it works: architecture](docs/architecture.md)**.

## Installation

**Prerequisites**: [uv](https://docs.astral.sh/uv/), Docker with Compose
(optional, only for Elasticsearch/Postgres/Grafana/Kestra/dlt), and an OpenAI
API key.

> **Tip:** every command below is a `make` target. Run `make help` at any
> time to see the full list with descriptions:
>
> ```
> $ make help
>
> ArXiv AI Papers RAG Chat
>
>   help         Show available targets
>   ensure-env   Create .env and make sure OPENAI_API_KEY is set (prompts if absent)
>   setup        Install pinned dependencies (uv sync) and configure .env
>   api          Start only the FastAPI backend on :$(API_PORT)
>   ui           Start only the Streamlit chat UI on :$(UI_PORT)
>   run          Start API + UI together (may take a few seconds to warm up)
>   infra        Start optional side infra: Elasticsearch + app Postgres
>   monitor      Start Postgres + Grafana for monitoring/feedback (needs 'make api' too)
>   ingestion    Start Postgres + dlt ingestion service + Kestra
>   eval         Build the index, start Elasticsearch, open the marimo evaluation notebook
>   down         Stop and remove docker compose containers (keeps named volumes)
>   stop         Stop docker compose containers (keeps them + volumes)
>   logs         Tail docker compose logs
> ```

One command installs the pinned dependencies **and** sets up `.env`:

```
make setup
```

`make setup` will:
1. Create `.env` from `.env.example` if it doesn't exist.
2. Check whether `OPENAI_API_KEY` is set. If not, it **prompts you to paste
   the key** (and writes it to `.env`), or aborts with instructions.
3. Run `uv sync` to install the pinned environment.

You can also do it manually:

```bash
uv sync
cp .env.example .env     # then set OPENAI_API_KEY=... in .env
```

Key variables (full commented list in [.env.example](.env.example)):

<details>
<summary>Table of variables :</summary>

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | **Required.** Used for embeddings and chat completions |
| `OPENAI_CHAT_MODEL` / `OPENAI_EMBEDDING_MODEL` | `gpt-4o-mini` / `text-embedding-3-small` | Models for generation / embeddings |
| `RETRIEVAL_STRATEGY` | `hybrid_rerank` | `hybrid_rerank` (recommended), `hybrid_only`, or `elasticsearch_rrf` (needs Elasticsearch) |
| `HYBRID_CANDIDATE_K` / `RERANK_TOP_K` / `RRF_K` | `100` / `5` / `60` | Hybrid search pool size, final context size, RRF constant |
| `ES_URL` | `http://localhost:9200` | Elasticsearch endpoint (only for `elasticsearch_rrf`) |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | `localhost:5432`, db/user/pass `arxiv_rag` | Conversation + feedback store |
| `BACKEND_URL` | `http://localhost:8000` | Where the Streamlit UI reaches the API |

</details>

### Building the index

Before first chat, build the index (embeds all papers, a one-time OpenAI
cost):

```
uv run python scripts/build_index.py
```

> [!NOTE]
> Use the `elasticsearch_rrf` strategy only for 'Retrieval Evaluation' or to test changing the `.env` 

The `elasticsearch_rrf` strategy, load Elasticsearch (docker-compose):

```
docker compose up -d elasticsearch    # rrf strategy to use with marimo or test on the app

# |   command  |  # check with python
uv run python -c "from arxiv_rag.indexing import get_index; from arxiv_rag.es_search import index_to_elasticsearch; index_to_elasticsearch(get_index())"
```

## Running the app

```
make run
```

Starts both the **API** (`http://localhost:8000`) and the **Streamlit chat UI**
(`http://localhost:8501`). Press `Ctrl+C` to stop both, or start them in
separate terminals:

```
make api    # uvicorn on :8000
make ui     # streamlit on :8501
```

Sanity-check the API: `curl http://localhost:8000/health`. The interactive
Scalar API client is at `http://localhost:8000/scalar`
(see [API & Scalar](docs/api_scalar.md)).

Optional side infrastructure:

```
make infra      # Elasticsearch + app Postgres (logging)
make monitor    # Postgres + Grafana :3000
make ingestion  # Kestra :8080 + dlt service :8090
make eval       # build index + open the marimo evaluation notebook
```

## Documentation

| Page | Contents |
|---|---|
| [Docs home](docs/README.md) | Index of all documentation pages |
| [How it works: architecture](docs/architecture.md) | Full RAG flow, retrieval strategies, prompts, cost tracking, indexing |
| [API & Scalar](docs/api_scalar.md) | Every FastAPI endpoint, request/response examples, Scalar client |
| [Retrieval evaluation](docs/retrieval_evaluation.md) | The marimo notebook comparing strategies, how to run it, latest results |
| [Monitoring & feedback](docs/monitoring_feedback.md) | Postgres feedback + the 11-panel Grafana dashboard |
| [Ingestion pipeline](docs/ingestion_pipeline.md) | dlt + DuckDB pipeline and the daily Kestra schedule |

## Tech Stack

- **Package Manager**: uv
- **Backend / API**: FastAPI, Uvicorn
- **Frontend / UI**: Streamlit
- **LLM**: OpenAI `gpt-4o-mini` (chat), `text-embedding-3-small` (embeddings)
- **Retrieval**: hybrid search — BM25 (`rank-bm25`) + dense embeddings (`sentence-transformers`), with cross-encoder re-ranking and RRF fusion
- **Search / Vector Store**: Elasticsearch, NumPy in-memory index
- **Database**: PostgreSQL (conversation logs, feedback), DuckDB (ingested dataset, via `dlt`)
- **Data Ingestion**: dlt (`dlt[duckdb]`), orchestrated daily by Kestra
- **Monitoring**: Grafana (11-panel dashboard over Postgres)
- **API Docs**: Scalar (`scalar-fastapi`)
- **Evaluation**: marimo notebooks
- **Data Validation**: Pydantic
- **Containerization**: Docker Compose
- **Code Quality**: ruff
