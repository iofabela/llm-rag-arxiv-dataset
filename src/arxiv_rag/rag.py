"""Orchestrates hybrid retrieval -> rerank -> grounded chat completion."""

import time

from pydantic import BaseModel

from .config import settings
from .indexing import RagIndex, get_index
from .openai_client import get_client
from .rerank import RankedResult
from .strategies import RETRIEVAL_STRATEGIES

# USD per 1M tokens (input, output). Unpriced models cost 0 rather than fail.
MODEL_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = MODEL_PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about AI/ML research papers. "
    "You are given a numbered list of retrieved arXiv papers (title, authors, publication date, "
    "and summary). Answer the user's question using ONLY the information in these papers -- do not "
    "use outside knowledge and do not invent facts, authors, or results. "
    "When you use information from a paper, cite it inline using its bracketed number, e.g. [1], "
    "or by title. If the provided papers do not contain enough information to answer the question, "
    "say so explicitly instead of guessing."
)

USER_PROMPT_TEMPLATE = "Retrieved papers:\n\n{context}\n\nQuestion: {question}"

NO_RESULTS_ANSWER = "I couldn't find any papers in the dataset relevant to your question."


class SourcePaper(BaseModel):
    entry_id: str
    title: str
    authors: str
    published: str
    pdf_url: str
    relevance_score: float


class RagAnswer(BaseModel):
    answer: str
    sources: list[SourcePaper]
    model: str
    retrieval_strategy: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float


def _format_published(value) -> str:
    return str(value)[:10]


def build_context_block(results: list[RankedResult]) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        row = result.row
        blocks.append(
            f"[{i}] Title: {row['title']}\n"
            f"Authors: {row['authors']}\n"
            f"Published: {_format_published(row['published'])}\n"
            f"arXiv: {row['entry_id']}\n"
            f"Summary: {row['summary']}"
        )
    return "\n\n".join(blocks)


def build_source_papers(results: list[RankedResult]) -> list[SourcePaper]:
    return [
        SourcePaper(
            entry_id=result.row["entry_id"],
            title=result.row["title"],
            authors=result.row["authors"],
            published=_format_published(result.row["published"]),
            pdf_url=result.row["pdf_url"],
            relevance_score=result.rerank_score,
        )
        for result in results
    ]


def answer_question(
    question: str,
    top_k: int | None = None,
    index: RagIndex | None = None,
    strategy: str | None = None,
) -> RagAnswer:
    start = time.perf_counter()

    idx = index or get_index()
    strategy_name = strategy or settings.retrieval_strategy
    strategy_fn = RETRIEVAL_STRATEGIES[strategy_name]

    ranked = strategy_fn(idx, question, top_k or settings.rerank_top_k)

    if not ranked:
        return RagAnswer(
            answer=NO_RESULTS_ANSWER,
            sources=[],
            model=settings.openai_chat_model,
            retrieval_strategy=strategy_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_time=time.perf_counter() - start,
            cost=0.0,
        )

    context = build_context_block(ranked)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    client = get_client()
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer_text = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    return RagAnswer(
        answer=answer_text,
        sources=build_source_papers(ranked),
        model=settings.openai_chat_model,
        retrieval_strategy=strategy_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        response_time=time.perf_counter() - start,
        cost=_compute_cost(settings.openai_chat_model, prompt_tokens, completion_tokens),
    )
