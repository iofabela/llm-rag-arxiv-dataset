"""Small HTTP wrapper used by the Kestra flow to trigger dlt."""

from __future__ import annotations

from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .dlt_pipeline import run_pipeline

app = FastAPI(title="arxiv-dlt-ingestion")
_run_lock = Lock()


class IngestionRequest(BaseModel):
    force_download: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run(request: IngestionRequest) -> dict[str, str]:
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingestion is already running")

    try:
        info = run_pipeline(force_download=request.force_download)
        return {
            "status": "success",
            "pipeline": "arxiv_ingestion",
            "load_info": str(info),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"dlt ingestion failed: {exc}") from exc
    finally:
        _run_lock.release()
