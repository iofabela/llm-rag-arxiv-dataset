"""Build (or load from cache) the dense embedding matrix and the BM25 sparse index."""

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .config import settings
from .data import load_processed_dataset
from .openai_client import get_client

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _embeddings_path() -> str:
    return os.path.join(settings.artifacts_dir, "embeddings.npy")


def _dataset_path() -> str:
    return os.path.join(settings.artifacts_dir, "processed_dataset.parquet")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts with the OpenAI embeddings API, batched, L2-normalized so
    cosine similarity reduces to a dot product against the stored matrix.
    """
    client = get_client()
    batch_size = settings.embedding_batch_size
    vectors: list[list[float]] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=settings.openai_embedding_model, input=batch)
        vectors.extend(item.embedding for item in response.data)

    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class RagIndex:
    df: pd.DataFrame
    embeddings: np.ndarray
    bm25: BM25Okapi


def build_or_load_index(force_rebuild: bool = False) -> RagIndex:
    df = load_processed_dataset()

    os.makedirs(settings.artifacts_dir, exist_ok=True)
    embeddings_path = _embeddings_path()
    dataset_path = _dataset_path()

    cache_valid = not force_rebuild and os.path.exists(embeddings_path) and os.path.exists(dataset_path)
    if cache_valid:
        cached_df = pd.read_parquet(dataset_path)
        embeddings = np.load(embeddings_path)
        if len(cached_df) == len(df) and len(embeddings) == len(df):
            df = cached_df
        else:
            cache_valid = False

    if not cache_valid:
        embeddings = embed_texts(df["index_text"].tolist())
        df.to_parquet(dataset_path, index=False)
        np.save(embeddings_path, embeddings)

    tokenized_corpus = [tokenize(text) for text in df["index_text"]]
    bm25 = BM25Okapi(tokenized_corpus)

    return RagIndex(df=df, embeddings=embeddings, bm25=bm25)


@lru_cache
def get_index() -> RagIndex:
    return build_or_load_index()
