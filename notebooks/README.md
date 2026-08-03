# Evaluation notebook

`retrieval_evaluation.py` is a [marimo](https://marimo.io) notebook that
compares the three retrieval strategies in `src/arxiv_rag/strategies.py`
(`hybrid_rerank`, `hybrid_only`, `elasticsearch_rrf`) on a sample of the
arXiv dataset -- retrieval quality (hit rate / MRR), answer relevancy
(LLM-as-a-judge), token cost, and latency -- and combines them into an
adjustable composite score. `eval_lib.py` is its supporting code (question
generation, judging, metrics); it's only imported by this notebook, not by
the production app.

## Prerequisites

1. **Dependencies installed** from the repo root:
   ```
   uv sync
   ```
2. **`.env` configured** at the repo root (`cp .env.example .env` if you
   haven't already), with at minimum `OPENAI_API_KEY` set. The notebook
   generates questions, answers, and LLM-judge scores via OpenAI chat
   completions (`gpt-4o-mini` by default) -- with `MAX_WORKERS = 3` and
   `SAMPLE_SIZE = 200` papers, a full run costs a few dollars and takes
   roughly 15-30 minutes, depending on OpenAI latency.
3. **The index built**, so `get_index()` can load cached embeddings instead
   of re-embedding the dataset:
   ```
   uv run python scripts/build_index.py
   ```
4. **Elasticsearch running**, for the `elasticsearch_rrf` strategy:
   ```
   docker compose up -d
   ```
   This starts the `arxiv-rag-elasticsearch` container defined in the
   repo-root `docker-compose.yml` (Elasticsearch 8.17, single-node, no
   auth, port 9200). Check it came up cleanly:
   ```
   docker compose ps
   curl http://localhost:9200
   ```
   The notebook's "Elasticsearch setup" cell calls `index_exists()`, which
   makes a real HTTP request to `ES_URL` (default
   `http://localhost:9200`) -- if nothing is listening there you'll see a
   `ConnectionError`. This is **not fatal**: the cell catches the
   exception and the notebook automatically drops `elasticsearch_rrf` from
   `strategy_names`, evaluating only `hybrid_rerank` and `hybrid_only`
   instead. Elasticsearch is only *required* if you want the
   `elasticsearch_rrf` strategy included in the comparison.
   - If the index doesn't exist yet, the notebook creates it and bulk-indexes
     the dataset automatically (reusing the embeddings already computed by
     `build_index.py` -- no re-embedding). If it already exists, it's reused
     as-is.
   - Docker isn't a hard requirement of the client itself -- `get_es_client()`
     just builds an HTTP client for whatever `ES_URL` points to. Docker is
     simply the convenient way this repo provisions a local ES server; a
     native ES install or a remote/Elastic Cloud cluster (via `ES_URL` /
     `ES_API_KEY` in `.env`) works too.
   - If the container was previously stopped (e.g. `Exited (137)`, an
     OOM-kill from the `-Xms1g -Xmx1g` heap in `docker-compose.yml` under a
     tight Docker Desktop memory limit), `docker compose up -d` restarts the
     existing container. Check `docker logs arxiv-rag-elasticsearch` if it
     keeps dying, and consider raising Docker Desktop's memory allocation.

## Running

Edit interactively (recommended -- lets you inspect intermediate cells,
tweak the composite-score weight sliders, and rerun individual cells):
```
uv run marimo edit notebooks/retrieval_evaluation.py
```

Or run it end-to-end from the CLI:
```
uv run marimo run notebooks/retrieval_evaluation.py
```

Either way, run it from the repo root (or ensure `notebooks/` is on
`PYTHONPATH`) so `import eval_lib` resolves.

## Output

Results and charts are written to `notebooks/artifacts/`:
- `eval_results.csv` -- raw per-(question, strategy) results
- `eval_retrieval_metrics.png`, `eval_cost.png`, `eval_latency.png`,
  `eval_relevancy.png`, `eval_composite_score.png`

The notebook's conclusion cell prints the winning strategy and the
`RETRIEVAL_STRATEGY=<winner>` line to set in `.env` to make the production
app (`src/arxiv_rag/rag.py`) use it -- no code changes needed.

result: Drafted notebooks/README.md content covering setup prerequisites (deps, .env, built index), the Elasticsearch/Docker requirement for the elasticsearch_rrf strategy (including the graceful-skip behavior and the OOM-kill gotcha we hit earlier), how to run the notebook, and where outputs land — provided above for you to save yourself, since this background session's isolation guard blocks direct writes to your live checkout.