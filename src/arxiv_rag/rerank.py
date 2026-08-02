"""Cross-encoder reranking of the hybrid-search candidates.

Runs locally via sentence-transformers (no extra API key, no extra network
call) since OpenAI has no dedicated rerank endpoint.
"""

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import settings
from .search import Candidate


@lru_cache
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.cross_encoder_model)


@dataclass
class RankedResult:
    row: object
    rerank_score: float


def rerank(query: str, candidates: list[Candidate], top_k: int | None = None) -> list[RankedResult]:
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []

    model = get_cross_encoder()
    pairs = [(query, candidate.row["index_text"]) for candidate in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        (RankedResult(row=candidate.row, rerank_score=float(score)) for candidate, score in zip(candidates, scores)),
        key=lambda result: result.rerank_score,
        reverse=True,
    )
    return ranked[:top_k]
