"""FastAPI app exposing the RAG chat backend."""

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import APIStatusError
from pydantic import BaseModel, Field
from scalar_fastapi import get_scalar_api_reference

from . import db
from .config import settings
from .indexing import get_index
from .rag import RagAnswer, answer_question
from .rerank import get_cross_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_index()
        get_cross_encoder()
    except Exception as exc:  # noqa: BLE001
        logger.warning("index warm-up deferred: %s", exc)
    try:
        db.init_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Postgres unavailable, conversation logging disabled: %s", exc)
    yield


app = FastAPI(
    title="arxiv-rag-api",
    description=(
        "Backend API for the ArXiv AI Papers RAG Chat project.\n\n"
        "Ask natural-language questions about ArXiv AI papers and get answers "
        "grounded in retrieved passages, using a hybrid search + reranking "
        "retrieval pipeline in front of an LLM. Conversations and user feedback "
        "are logged to Postgres for evaluation and monitoring (see the Grafana "
        "dashboard)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/scalar", include_in_schema=False)
def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatResponse(RagAnswer):
    # None when Postgres is unreachable -- chat still works, feedback just can't be filed.
    conversation_id: int | None = None


class FeedbackRequest(BaseModel):
    conversation_id: int
    feedback: Literal[-1, 1]


def _openai_error_detail(exc: APIStatusError) -> tuple[int, str]:
    """Map an OpenAI API error to a safe (status_code, detail) pair for clients.

    Never forwards the raw provider error body to the client -- that's for the
    server log only. Returns 429 for rate limits/quota issues (so the frontend
    can tell "try again shortly" apart from "the server is broken"), and 502
    for anything else the upstream API rejected us for.
    """
    if exc.status_code == 429:
        if exc.code in ("insufficient_quota", "credit_balance_exhausted") or exc.type == "insufficient_quota":
            return 429, (
                "The AI provider account has run out of credits. "
                "Please contact the site administrator to top up billing and try again later."
            )
        return 429, "The AI provider is receiving too many requests right now. Please wait a moment and try again."

    if exc.status_code in (401, 403):
        return 502, "The backend is misconfigured (AI provider credentials rejected). Please contact the site administrator."

    return 502, "The AI provider is temporarily unavailable. Please try again later."


@app.get("/health", summary="Health check")
def health() -> dict:
    """Report service liveness plus whether OpenAI is configured and the
    retrieval index has been warmed up in memory."""
    return {
        "status": "ok",
        "openai_configured": bool(settings.openai_api_key),
        "index_loaded": get_index.cache_info().currsize > 0,
    }


@app.post("/chat", response_model=ChatResponse, summary="Ask a question")
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question about ArXiv AI papers.

    Retrieves relevant passages via the configured retrieval strategy,
    reranks them, and generates an answer with an LLM. The response includes
    usage/cost metadata and a `conversation_id` that can be used to submit
    feedback via `POST /feedback` (null if Postgres logging is unavailable).
    """
    try:
        result = answer_question(request.question, top_k=request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except APIStatusError as exc:
        logger.error(
            "OpenAI API error while answering question %r: status=%s code=%s type=%s body=%s",
            request.question,
            exc.status_code,
            exc.code,
            exc.type,
            exc.body,
        )
        status_code, detail = _openai_error_detail(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error answering question %r", request.question)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while answering the question.") from exc

    conversation_id = None
    try:
        conversation_id = db.save_conversation(
            question=request.question,
            answer=result.answer,
            model=result.model,
            retrieval_strategy=result.retrieval_strategy,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            response_time=result.response_time,
            cost=result.cost,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[chat] failed to log conversation to Postgres: {exc}")

    return ChatResponse(**result.model_dump(), conversation_id=conversation_id)


@app.post("/feedback", summary="Rate an answer")
def feedback(request: FeedbackRequest) -> dict:
    """Record a thumbs up (`1`) or thumbs down (`-1`) for a previous chat
    answer, identified by the `conversation_id` returned from `POST /chat`."""
    try:
        db.save_feedback(request.conversation_id, request.feedback)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {exc}") from exc
    return {"status": "ok"}
