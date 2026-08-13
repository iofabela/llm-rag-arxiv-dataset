# ArXiv AI Papers RAG Chat — Makefile
#
# Convenience wrappers around uv / docker compose / marimo. The app itself
# (FastAPI + Streamlit) runs locally via uv; docker compose only starts the
# optional infrastructure (Elasticsearch, Postgres, Grafana, Kestra, dlt).
#
# Typical first run:
#   make setup     # installs deps and ensures .env / OPENAI_API_KEY
#   make run       # starts the API + chat UI, Ctrl+C stops both

SHELL := /bin/bash

API_HOST ?= 0.0.0.0
API_PORT ?= 8000
UI_PORT  ?= 8501

export API_HOST API_PORT UI_PORT

.PHONY: help setup ensure-env run api ui infra monitor ingestion eval down stop logs

help: ## Show available targets
	@printf '\n\033[1mArXiv AI Papers RAG Chat\033[0m\n\n'
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@printf '\n'

## ---- environment bootstrap ------------------------------------------------

ensure-env: ## Create .env and make sure OPENAI_API_KEY is set (prompts if absent)
	@bash scripts/ensure_env.sh

setup: ensure-env ## Install pinned dependencies (uv sync) and configure .env
	uv sync
	@printf '\n\033[1;32mDone.\033[0m Next: \033[1mmake run\033[0m to start the app, or \033[1mmake help\033[0m.\n'

## ---- app (FastAPI + Streamlit), run locally via uv -------------------------

api: ensure-env ## Start only the FastAPI backend on :$(API_PORT)
	uv run uvicorn arxiv_rag.api:app --host $(API_HOST) --port $(API_PORT) --reload

ui: ensure-env ## Start only the Streamlit chat UI on :$(UI_PORT)
	uv run streamlit run app_streamlit.py --server.port $(UI_PORT)

run: ensure-env ## Start API + UI together (may take a few seconds to warm up)
	@bash scripts/run_app.sh

## ---- side infrastructure (docker compose) -----------------------------------

infra: ## Start optional side infra: Elasticsearch + app Postgres
	docker compose up -d elasticsearch rag-postgres

monitor: ## Start Postgres + Grafana for monitoring/feedback (needs 'make api' too)
	docker compose up -d rag-postgres grafana
	@printf '\nGrafana: http://localhost:3000 (admin/admin).\n'

ingestion: ## Start Postgres + dlt ingestion service + Kestra
	docker compose up -d postgres dlt-ingestion kestra
	@printf '\nKestra: http://localhost:8080. dlt service: http://localhost:8090/health\n'

## ---- evaluation notebook (marimo) --------------------------------------------

eval: ensure-env ## Build the index, start Elasticsearch, open the marimo evaluation notebook
	uv run python scripts/build_index.py
	docker compose up -d elasticsearch
	uv run marimo edit notebooks/retrieval_evaluation.py

## ---- housekeeping -------------------------------------------------------------

down: ## Stop and remove docker compose containers (keeps named volumes)
	docker compose down

stop: ## Stop docker compose containers (keeps them + volumes)
	docker compose stop

logs: ## Tail docker compose logs
	docker compose logs -f