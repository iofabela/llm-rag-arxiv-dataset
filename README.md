# llm-rag-arxiv-dataset

Chat with a LLM-RAG over an arXiv AI research papers dataset.

## Architecture

- **Dataset**: [`yasirabdaali/arxivorg-ai-research-papers-dataset`](https://www.kaggle.com/datasets/yasirabdaali/arxivorg-ai-research-papers-dataset) (10,000 papers), fetched via `kagglehub` and cached to `data/arxiv_ai.csv`.
- **Ingestion**: dlt loads the cleaned dataset into `data/arxiv_papers.duckdb`. Kestra calls the dlt ingestion service daily and serializes runs so two refreshes cannot write the same DuckDB file concurrently.
- **Indexing**: each paper is indexed on `title + authors + summary`. Dense embeddings come from the OpenAI embeddings API (`text-embedding-3-small`), stored as an in-memory NumPy matrix (cached to `artifacts/`); a BM25 sparse index is rebuilt in memory on load.
- **Retrieval**: `elasticsearch_rrf` (Elasticsearch's native RRF retriever, fusing BM25 + kNN server-side) is the default retrieval strategy -- see [notebooks/retrieval_evaluation.py](notebooks/retrieval_evaluation.py) for why. Two alternate strategies are still available and fully implemented in `src/arxiv_rag/strategies.py`, selectable via `RETRIEVAL_STRATEGY` in `.env`: `hybrid_rerank` (NumPy/BM25 hybrid search + local cross-encoder rerank) and `hybrid_only` (same hybrid search, no rerank).
- **Generation**: the top retrieved papers are passed as context to an OpenAI chat completion (`gpt-4o-mini` by default), which answers grounded in those papers and cites them.
- **Backend**: FastAPI (`POST /chat`, `GET /health`).
- **Frontend**: Streamlit chat UI, calling the FastAPI backend over HTTP (an Astro frontend may replace/join it later, against the same API).

## Project layout

- `src/arxiv_rag/` -- the app package (everything `api.py` / `app_streamlit.py` actually run: config, data loading, indexing, the three retrieval strategies, generation, the FastAPI app).
- `app_streamlit.py`, `scripts/build_index.py` -- app entrypoints, at the repo root.
- `ingestion/` -- the dlt pipeline and HTTP service invoked by Kestra.
- `flows/arxiv_ingestion.yml` -- the daily Kestra flow, loaded automatically by Docker Compose.
- `notebooks/` -- evaluation only, not used by the running app: `retrieval_evaluation.py` (the marimo notebook that compares the retrieval strategies) and `eval_lib.py` (its supporting code -- question generation, LLM-as-a-judge, cost/latency metrics).
- `docker-compose.yml` -- local Elasticsearch, Kestra, its PostgreSQL metadata store, and the dlt ingestion service.

## Setup

1. Install dependencies:
   ```
   uv sync
   ```
2. Configure environment:
   ```
   cp .env.example .env
   ```
   Fill in `OPENAI_API_KEY` (required). Only fill in `KAGGLE_USERNAME`/`KAGGLE_KEY` if `data/arxiv_ai.csv` isn't already present locally and needs to be downloaded via kagglehub.
3. Start Elasticsearch, Kestra, PostgreSQL, and the dlt ingestion service:
    ```
    docker compose up -d --build
    ```
   Kestra is available at `http://localhost:8080`. The `arxiv.rag/arxiv_ingestion` flow is loaded from `flows/` and runs daily. Trigger it manually from the Kestra UI, or load the local CSV once with:
    ```
    curl -X POST http://localhost:8090/run \
      -H 'Content-Type: application/json' \
      -d '{"force_download": false}'
    ```
   Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` before using the scheduled refresh, which passes `force_download: true`.
4. Build the index (embeds all papers, one-time OpenAI cost) and populate Elasticsearch:
    ```
    uv run python scripts/build_index.py
    uv run python -c "from arxiv_rag.indexing import get_index; from arxiv_rag.es_search import index_to_elasticsearch; index_to_elasticsearch(get_index())"
    ```

The RAG loader reads the dlt-managed DuckDB table when it exists and falls back to `data/arxiv_ai.csv` otherwise. After a scheduled refresh, run `uv run python scripts/build_index.py --force-rebuild` and then the Elasticsearch load command to make changed papers searchable.

For a local dlt run without Kestra:

```
uv run python -m ingestion.dlt_pipeline
uv run python -m ingestion.dlt_pipeline --force-download
```

## Run

Start the API:
```
uv run uvicorn arxiv_rag.api:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal, start the chat UI:
```
uv run streamlit run app_streamlit.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Evaluating retrieval strategies

```
uv run marimo edit notebooks/retrieval_evaluation.py
```

Compares `hybrid_rerank`, `hybrid_only`, and `elasticsearch_rrf` on a 200-paper sample: retrieval quality (hit rate / MRR), answer relevancy (LLM-as-a-judge), token cost, and latency, combined into an adjustable composite score. Change `RETRIEVAL_STRATEGY` in `.env` to switch which one the running app uses -- no code changes needed.
