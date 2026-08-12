# ArXiv Ingestion

This directory contains the dlt pipeline and the HTTP service used by Kestra.
The pipeline loads the cleaned arXiv dataset into DuckDB at
`data/arxiv_papers.duckdb`.

## Process

1. Read `data/arxiv_ai.csv`.
2. Download the Kaggle snapshot first when `force_download` is enabled.
3. Remove discarded columns and create the shared `index_text` field.
4. Load the records with dlt into the `arxiv_data.arxiv_papers` DuckDB table.
5. Rebuild the RAG artifacts with `scripts/build_index.py --force-rebuild` when changed papers should become searchable.

The dlt resource uses `replace`, so each successful run creates a complete,
consistent refresh instead of appending duplicate papers.

## Configuration

Run commands from the repository root. Copy `.env.example` to `.env` if this
has not been done already.

The checked-in CSV is enough for a local run. Set `KAGGLE_USERNAME` and
`KAGGLE_KEY` in `.env` when the pipeline must download a fresh Kaggle snapshot.

Important settings are:

- `DATA_DIR`: source and destination data directory.
- `RAW_CSV_FILENAME`: source CSV name.
- `DLT_DATABASE_PATH`: DuckDB destination path.
- `DLT_DATASET_NAME`: DuckDB schema name.
- `DLT_RESOURCE_NAME`: DuckDB table name.

## Run Locally

Load the existing CSV without downloading:

```bash
uv run python -m ingestion.dlt_pipeline
```

Download the latest Kaggle snapshot and load it:

```bash
uv run python -m ingestion.dlt_pipeline --force-download
```

The pipeline prints dlt load metadata when it finishes. Verify the output
from Python if needed:

```bash
uv run python -c "from arxiv_rag.data import fetch_dlt_dataset; print(len(fetch_dlt_dataset()))"
```

## Run With Docker

Start the complete local stack, including Elasticsearch, PostgreSQL, Kestra,
and the dlt service:

```bash
docker compose up -d --build
```

To start only the dlt HTTP service:

```bash
docker compose up -d --build dlt-ingestion
```

Check the service:

```bash
curl http://localhost:8090/health
```

Trigger an ingestion from the checked-in CSV:

```bash
curl -X POST http://localhost:8090/run \
  -H 'Content-Type: application/json' \
  -d '{"force_download": false}'
```

Trigger a fresh Kaggle download:

```bash
curl -X POST http://localhost:8090/run \
  -H 'Content-Type: application/json' \
  -d '{"force_download": true}'
```

Follow service logs:

```bash
docker compose logs -f dlt-ingestion
```

## Kestra

Kestra is available at `http://localhost:8080`. The flow in
`flows/arxiv_ingestion.yml` is loaded when the Kestra container starts.

- Namespace: `arxiv.rag`
- Flow: `arxiv_ingestion`
- Schedule: daily at midnight UTC
- Scheduled runs set `force_download: true`
- Concurrent runs are limited to one

The flow calls `http://dlt-ingestion:8090/run` over the Docker Compose network.
For a manual run without downloading, execute the flow from the Kestra UI and
leave `force_download` set to `false`.

## Stop Containers

Stop only the ingestion service while preserving its container and volumes:

```bash
docker compose stop dlt-ingestion
```

Stop the complete stack while preserving named volumes:

```bash
docker compose stop
```

Stop and remove the containers and Compose network while preserving named
volumes:

```bash
docker compose down
```

Do not use `docker compose down -v` unless the persisted data is intentionally
being deleted. That command removes the dlt state, Kestra state, PostgreSQL
metadata, and Elasticsearch data volumes.
