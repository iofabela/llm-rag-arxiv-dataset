# llm-rag-arxiv-dataset

Chat with a LLM-RAG over an arXiv AI research papers dataset.

## Architecture

- **Dataset**: [`yasirabdaali/arxivorg-ai-research-papers-dataset`](https://www.kaggle.com/datasets/yasirabdaali/arxivorg-ai-research-papers-dataset) (10,000 papers), fetched via `kagglehub` and cached to `data/arxiv_ai.csv`.
- **Indexing**: each paper is indexed on `title + authors + summary`. Dense embeddings come from the OpenAI embeddings API (`text-embedding-3-small`), stored as an in-memory NumPy matrix (cached to `artifacts/`); a BM25 sparse index is rebuilt in memory on load.
- **Retrieval**: hybrid search fuses the dense and sparse rankings with Reciprocal Rank Fusion, then a local sentence-transformers cross-encoder reranks the candidates (OpenAI has no native rerank endpoint).
- **Generation**: the top reranked papers are passed as context to an OpenAI chat completion (`gpt-4o-mini` by default), which answers grounded in those papers and cites them.
- **Backend**: FastAPI (`POST /chat`, `GET /health`).
- **Frontend**: Streamlit chat UI, calling the FastAPI backend over HTTP (an Astro frontend may replace/join it later, against the same API).

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
3. Build the index (embeds all papers, one-time OpenAI cost):
   ```
   uv run python scripts/build_index.py
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
