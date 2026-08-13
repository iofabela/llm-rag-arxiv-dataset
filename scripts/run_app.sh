#!/usr/bin/env bash
# Start the FastAPI backend + Streamlit UI together; Ctrl+C stops both.
set -euo pipefail

API_HOST=${API_HOST:-0.0.0.0}
API_PORT=${API_PORT:-8000}
UI_PORT=${UI_PORT:-8501}

pids=()

cleanup() {
  # The CLI delivers Ctrl+C to the whole foreground process group, but if a
  # graceful exit is triggered only in this wrapper, forward TERM to children.
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

uv run uvicorn arxiv_rag.api:app --host "$API_HOST" --port "$API_PORT" --reload &
pids+=("$!")

uv run streamlit run app_streamlit.py --server.port "$UI_PORT" &
pids+=("$!")

echo "[run] API    → http://localhost:$API_PORT (Scalar at /scalar)"
echo "[run] UI     → http://localhost:$UI_PORT"
echo "[run] Ctrl+C to stop both."

wait