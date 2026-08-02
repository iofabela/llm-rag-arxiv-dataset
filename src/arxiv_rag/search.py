"""Hybrid retrieval: fuse BM25 (sparse) and OpenAI embedding (dense) rankings
with Reciprocal Rank Fusion (RRF).

RRF is used instead of a weighted score blend because BM25 scores and cosine
similarities live on different, unbounded/bounded scales that shift with the
query -- blending them directly requires constant re-tuning. RRF only needs
each retriever's *rank order*, so it combines them robustly with one stable
constant (k).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import settings
from .indexing import RagIndex, tokenize
from .openai_client import get_client


def _dense_ranking(index: RagIndex, query_embedding: np.ndarray, top_k: int) -> list[int]:
    scores = index.embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:top_k]
    return top_indices.tolist()


def _sparse_ranking(index: RagIndex, query: str, top_k: int) -> list[int]:
    scores = index.bm25.get_scores(tokenize(query))
    top_indices = np.argsort(-scores)[:top_k]
    return top_indices.tolist()


def _embed_query(query: str) -> np.ndarray:
    client = get_client()
    response = client.embeddings.create(model=settings.openai_embedding_model, input=[query])
    vector = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Combine multiple rank-ordered document-index lists into one fused
    ranking. Returns (doc_index, fused_score) sorted best-first.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_index in enumerate(ranking):
            scores[doc_index] = scores.get(doc_index, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


@dataclass
class Candidate:
    row: pd.Series
    fused_score: float


def hybrid_search(index: RagIndex, query: str, top_k: int | None = None) -> list[Candidate]:
    top_k = top_k or settings.hybrid_candidate_k

    query_embedding = _embed_query(query)
    dense_ranking = _dense_ranking(index, query_embedding, top_k)
    sparse_ranking = _sparse_ranking(index, query, top_k)

    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=settings.rrf_k)

    return [Candidate(row=index.df.iloc[doc_index], fused_score=score) for doc_index, score in fused[:top_k]]
