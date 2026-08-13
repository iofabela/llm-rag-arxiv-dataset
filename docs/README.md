# Documentation — ArXiv AI Papers RAG Chat

Supplementary pages to the [root README](../README.md). The README gives the
big picture (problem, data, summarized flow, quick install); these pages go
deeper into each subsystem.

| Page | Topics |
|---|---|
| [architecture.md](architecture.md) | Full RAG flow: UI → API → retrieval strategies → generation → logging/cost → indexing |
| [api_scalar.md](api_scalar.md) | FastAPI endpoints and the interactive Scalar API client |
| [retrieval_evaluation.md](retrieval_evaluation.md) | The marimo evaluation notebook: how strategies are compared, how to run it, results |
| [monitoring_feedback.md](monitoring_feedback.md) | 👍/👎 feedback persisted in Postgres and the Grafana dashboard |
| [ingestion_pipeline.md](ingestion_pipeline.md) | The dlt + DuckDB pipeline and the daily Kestra schedule |

Reference code for each area:

- API and retrieval strategies: [`src/arxiv_rag/`](../src/arxiv_rag)
- Evaluation harness: [`notebooks/`](../notebooks)
- Grafana dashboard: [`grafana/provisioning/`](../grafana/provisioning)
- Ingestion: [`ingestion/`](../ingestion) and [`flows/arxiv_ingestion.yml`](../flows/arxiv_ingestion.yml)