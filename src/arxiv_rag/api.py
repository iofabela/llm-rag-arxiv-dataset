"""FastAPI app exposing the RAG chat backend."""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import db
from .config import settings
from .indexing import get_index
from .rag import RagAnswer, answer_question
from .rerank import get_cross_encoder


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_index()
        get_cross_encoder()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] index warm-up deferred: {exc}")
    try:
        db.init_db()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] Postgres unavailable, conversation logging disabled: {exc}")
    yield


app = FastAPI(title="arxiv-rag-api", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "openai_configured": bool(settings.openai_api_key),
        "index_loaded": get_index.cache_info().currsize > 0,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = answer_question(request.question, top_k=request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc

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


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    try:
        db.save_feedback(request.conversation_id, request.feedback)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {exc}") from exc
    return {"status": "ok"}
