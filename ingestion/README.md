# ArXiv Ingestion

This directory contains the dlt pipeline (`dlt_pipeline.py`) and the HTTP
service (`service.py`) used by Kestra. The pipeline loads the cleaned arXiv
dataset into DuckDB at `data/arxiv_papers.duckdb`.

Full documentation is in **[docs/ingestion_pipeline.md](../docs/ingestion_pipeline.md)**:

- the process and configuration
- local and Docker runs (`make ingestion`)
- the dlt HTTP service endpoints (`GET /health`, `POST /run`)
- the Kestra daily schedule