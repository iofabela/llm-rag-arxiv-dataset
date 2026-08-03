"""Elasticsearch-backed hybrid retrieval, using ES's native RRF retriever
(server-side fusion of a BM25 match query and a kNN dense-vector query).

Reuses the same cached embeddings as the NumPy/BM25 pipeline (see
`indexing.build_or_load_index`) so all retrieval approaches compare against
identical vectors -- only the retrieval mechanism differs.
"""

from functools import lru_cache

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from .config import settings
from .indexing import RagIndex
from .search import Candidate, _embed_query

_TEXT_FIELDS = ["index_text", "title", "authors", "summary", "categories", "primary_category", "published", "entry_id", "pdf_url", "journal_ref", "updated"]


@lru_cache
def get_es_client() -> Elasticsearch:
    kwargs = {}
    if settings.es_api_key:
        kwargs["api_key"] = settings.es_api_key
    return Elasticsearch(settings.es_url, **kwargs)


def index_exists() -> bool:
    return get_es_client().indices.exists(index=settings.es_index_name)


def index_to_elasticsearch(index: RagIndex, recreate: bool = False) -> None:
    """Create the ES index (if missing) and bulk-index the dataset, reusing
    the already-computed embedding matrix rather than re-embedding.
    """
    client = get_es_client()
    dims = index.embeddings.shape[1]

    if recreate and client.indices.exists(index=settings.es_index_name):
        client.indices.delete(index=settings.es_index_name)

    if not client.indices.exists(index=settings.es_index_name):
        client.indices.create(
            index=settings.es_index_name,
            mappings={
                "properties": {
                    "index_text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "entry_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "pdf_url": {"type": "keyword"},
                }
            },
        )

    def actions():
        for i, row in index.df.iterrows():
            source = {field: (None if pd.isna(row[field]) else row[field]) for field in _TEXT_FIELDS if field in row}
            source["embedding"] = index.embeddings[i].tolist()
            yield {"_index": settings.es_index_name, "_id": row["entry_id"], "_source": source}

    bulk(client, actions())
    client.indices.refresh(index=settings.es_index_name)


def hybrid_search_es(query: str, top_k: int) -> list[Candidate]:
    """Hybrid search via Elasticsearch's RRF retriever: BM25 match on
    `index_text` fused server-side with a kNN search over `embedding`.
    """
    client = get_es_client()
    query_embedding = _embed_query(query)

    response = client.search(
        index=settings.es_index_name,
        retriever={
            "rrf": {
                "retrievers": [
                    {"standard": {"query": {"match": {"index_text": query}}}},
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_embedding.tolist(),
                            "k": top_k,
                            "num_candidates": max(100, top_k * 10),
                        }
                    },
                ],
                "rank_window_size": max(50, top_k * 10),
                "rank_constant": settings.rrf_k,
            }
        },
        size=top_k,
    )

    candidates = []
    for hit in response["hits"]["hits"]:
        source = dict(hit["_source"])
        source.pop("embedding", None)
        candidates.append(Candidate(row=pd.Series(source), fused_score=hit["_score"]))
    return candidates
