"""Evaluation harness comparing retrieval strategies for the RAG pipeline.

Used only by notebooks/retrieval_evaluation.py -- not part of the production
app (api.py / rag.py / app_streamlit.py never import this). Kept alongside
the notebook rather than inside the arxiv_rag package to keep "what runs the
chat app" and "what evaluates retrieval strategies" clearly separate.

Strategies compared (see arxiv_rag.strategies, shared with production):
- hybrid_rerank:     NumPy/BM25 hybrid search (RRF) + local cross-encoder rerank
- hybrid_only:       NumPy/BM25 hybrid search (RRF) only, no rerank
- elasticsearch_rrf: Elasticsearch's native RRF retriever (BM25 + kNN), no rerank -- production default

All three produce a uniform list[RankedResult] so the same generation prompt
(rag.SYSTEM_PROMPT / rag.USER_PROMPT_TEMPLATE) is used downstream -- retrieval
method is the only thing that varies between strategies.
"""

import time
from dataclasses import dataclass

import pandas as pd
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from arxiv_rag.config import settings
from arxiv_rag.indexing import RagIndex
from arxiv_rag.openai_client import get_client
from arxiv_rag.rag import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, NO_RESULTS_ANSWER, build_context_block
from arxiv_rag.rerank import RankedResult
from arxiv_rag.strategies import RETRIEVAL_STRATEGIES

# The eval loop fires many concurrent chat-completion calls; a 429 just means
# "back off and retry", not a real failure -- retry with jittered backoff
# instead of letting one rate-limited call crash the whole run.
retry_on_rate_limit = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(6),
)

# Standard (non-batch) OpenAI pricing, USD per 1M tokens. Confirmed against
# https://developers.openai.com/api/docs/pricing on 2026-08-02 -- re-check
# before trusting cost figures if models/prices have since changed.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def calc_total_price(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    rates = PRICING.get(model)
    if rates is None:
        raise ValueError(f"No pricing data for model {model!r}; add it to PRICING.")
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


@dataclass
class UsageStat:
    model: str
    input_tokens: int
    output_tokens: int = 0

    @property
    def cost(self) -> float:
        return calc_total_price(self.model, self.input_tokens, self.output_tokens)


# --- Ground-truth question generation ---------------------------------------

QUESTION_GEN_SYSTEM_PROMPT = (
    """
    You write a single, natural user question that is fully answerable using ONLY the
    given paper's title, authors, and summary. The question should read like something
    a researcher would type into a search/chat tool -- specific enough that this paper is
    clearly the best answer, but do not quote the title verbatim. Return ONLY the question,
    no preamble, no quotes.
    """
)


@retry_on_rate_limit
def generate_question(row: pd.Series) -> tuple[str, UsageStat]:
    client = get_client()
    prompt = f"Title: {row['title']}\nAuthors: {row['authors']}\nSummary: {row['summary']}"
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": QUESTION_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    question = (response.choices[0].message.content or "").strip()
    usage = UsageStat(settings.openai_chat_model, response.usage.prompt_tokens, response.usage.completion_tokens)
    return question, usage


# --- Generation & judging ------------------------------------------------------


@retry_on_rate_limit
def generate_answer(question: str, ranked: list[RankedResult]) -> tuple[str, UsageStat]:
    if not ranked:
        return NO_RESULTS_ANSWER, UsageStat(settings.openai_chat_model, 0, 0)

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
    answer = response.choices[0].message.content or ""
    usage = UsageStat(settings.openai_chat_model, response.usage.prompt_tokens, response.usage.completion_tokens)
    return answer, usage


JUDGE_SYSTEM_PROMPT = (
    "You are grading how relevant an answer is to a question, on a 1-5 integer scale "
    "(5 = directly and completely answers the question; 1 = irrelevant or a non-answer). "
    "Respond with ONLY the integer score, nothing else."
)


@retry_on_rate_limit
def judge_relevancy(question: str, answer: str) -> tuple[int, UsageStat]:
    client = get_client()
    prompt = f"Question: {question}\n\nAnswer: {answer}"
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw = (response.choices[0].message.content or "").strip()
    score = int(raw[0]) if raw and raw[0].isdigit() else 1
    usage = UsageStat(settings.openai_chat_model, response.usage.prompt_tokens, response.usage.completion_tokens)
    return score, usage


# --- Retrieval metrics ----------------------------------------------------------


def retrieval_rank(ranked: list[RankedResult], target_entry_id: str) -> int | None:
    for i, result in enumerate(ranked, start=1):
        if result.row["entry_id"] == target_entry_id:
            return i
    return None


def hit_rate(ranks: list[int | None]) -> float:
    return sum(1 for r in ranks if r is not None) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    return sum((1 / r) if r is not None else 0.0 for r in ranks) / len(ranks)


# --- Per-question evaluation ----------------------------------------------------


@dataclass
class EvalResult:
    entry_id: str
    question: str
    strategy: str
    rank: int | None
    answer: str
    relevancy: int
    input_tokens: int
    output_tokens: int
    cost: float
    retrieval_latency_s: float
    total_latency_s: float


def evaluate_one(index: RagIndex, entry_id: str, question: str, strategy_name: str, top_k: int) -> EvalResult:
    strategy_fn = RETRIEVAL_STRATEGIES[strategy_name]

    start = time.time()
    ranked = strategy_fn(index, question, top_k)
    retrieval_latency = time.time() - start

    rank = retrieval_rank(ranked, entry_id)
    answer, gen_usage = generate_answer(question, ranked)
    relevancy, judge_usage = judge_relevancy(question, answer)
    total_latency = time.time() - start

    return EvalResult(
        entry_id=entry_id,
        question=question,
        strategy=strategy_name,
        rank=rank,
        answer=answer,
        relevancy=relevancy,
        input_tokens=gen_usage.input_tokens + judge_usage.input_tokens,
        output_tokens=gen_usage.output_tokens + judge_usage.output_tokens,
        cost=gen_usage.cost + judge_usage.cost,
        retrieval_latency_s=retrieval_latency,
        total_latency_s=total_latency,
    )


# --- Aggregation & scoring ---------------------------------------------------


def summarize(results: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for strategy, group in results.groupby("strategy"):
        ranks = group["rank"].tolist()
        rows.append(
            {
                "strategy": strategy,
                f"hit_rate@{top_k}": hit_rate(ranks),
                f"mrr@{top_k}": mean_reciprocal_rank(ranks),
                "avg_relevancy": group["relevancy"].mean(),
                "avg_retrieval_latency_s": group["retrieval_latency_s"].mean(),
                "avg_total_latency_s": group["total_latency_s"].mean(),
                "total_cost_usd": group["cost"].sum(),
                "avg_cost_usd": group["cost"].mean(),
                "total_input_tokens": int(group["input_tokens"].sum()),
                "total_output_tokens": int(group["output_tokens"].sum()),
            }
        )
    return pd.DataFrame(rows)


DEFAULT_SCORE_WEIGHTS = {"retrieval": 0.30, "relevancy": 0.35, "speed": 0.15, "cost": 0.20}


def compute_composite_score(summary: pd.DataFrame, top_k: int, weights: dict | None = None) -> pd.DataFrame:
    weights = weights or DEFAULT_SCORE_WEIGHTS
    df = summary.copy()

    def norm(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series([1.0] * len(series), index=series.index)
        n = (series - lo) / (hi - lo)
        return n if higher_is_better else 1 - n

    retrieval_quality = (norm(df[f"hit_rate@{top_k}"]) + norm(df[f"mrr@{top_k}"])) / 2
    relevancy_norm = norm(df["avg_relevancy"])
    speed_norm = norm(df["avg_total_latency_s"], higher_is_better=False)
    cost_norm = norm(df["avg_cost_usd"], higher_is_better=False)

    df["score"] = (
        weights["retrieval"] * retrieval_quality
        + weights["relevancy"] * relevancy_norm
        + weights["speed"] * speed_norm
        + weights["cost"] * cost_norm
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)
