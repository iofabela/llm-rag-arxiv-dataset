"""Retrieval strategies, as a uniform interface: (index, query, top_k) -> list[RankedResult].

Shared by the production RAG pipeline (rag.py, strategy chosen via
settings.retrieval_strategy) and the evaluation harness (eval.py, which
compares all of them) so both use the exact same retrieval code.
"""

from .es_search import hybrid_search_es
from .indexing import RagIndex
from .rerank import RankedResult, rerank
from .search import hybrid_search


def strategy_hybrid_rerank(index: RagIndex, query: str, top_k: int) -> list[RankedResult]:
    """NumPy/BM25 hybrid search (RRF) + local cross-encoder rerank."""
    candidates = hybrid_search(index, query)
    return rerank(query, candidates, top_k=top_k)


def strategy_hybrid_only(index: RagIndex, query: str, top_k: int) -> list[RankedResult]:
    """NumPy/BM25 hybrid search (RRF) only, no rerank.

    Uses hybrid_search's default candidate pool (settings.hybrid_candidate_k),
    same as strategy_hybrid_rerank, and just truncates to top_k without
    reranking -- so the only variable between the two strategies is the
    presence of the rerank step, not the candidate pool size.
    """
    candidates = hybrid_search(index, query)
    return [RankedResult(row=c.row, rerank_score=c.fused_score) for c in candidates[:top_k]]


def strategy_elasticsearch_rrf(index: RagIndex, query: str, top_k: int) -> list[RankedResult]:
    """Elasticsearch's native RRF retriever (BM25 + kNN), no rerank."""
    candidates = hybrid_search_es(query, top_k=top_k)
    return [RankedResult(row=c.row, rerank_score=c.fused_score) for c in candidates]


RETRIEVAL_STRATEGIES = {
    "hybrid_rerank": strategy_hybrid_rerank,
    "hybrid_only": strategy_hybrid_only,
    "elasticsearch_rrf": strategy_elasticsearch_rrf,
}
