"""Cross-encoder reranking of the hybrid-search candidates.

Runs locally via sentence-transformers (no extra API key, no extra network
call) since OpenAI has no dedicated rerank endpoint.
"""

import threading
from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import settings
from .search import Candidate

# Concurrent CrossEncoder.predict() calls from multiple threads are not safe
# in PyTorch's native code (observed segfaults under ThreadPoolExecutor-based
# evaluation) -- serialize access to the model with a lock rather than
# skipping concurrency for the (I/O-bound) rest of the pipeline.
_predict_lock = threading.Lock()


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
    with _predict_lock:
        scores = model.predict(pairs)

    ranked = sorted(
        (RankedResult(row=candidate.row, rerank_score=float(score)) for candidate, score in zip(candidates, scores)),
        key=lambda result: result.rerank_score,
        reverse=True,
    )
    return ranked[:top_k]
