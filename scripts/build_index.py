"""Eagerly build/cache the dataset + embeddings index.

Usage:
    uv run python scripts/build_index.py [--force-download] [--force-rebuild]
"""

import argparse

from arxiv_rag.data import fetch_raw_dataset
from arxiv_rag.indexing import build_or_load_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or refresh the RAG index cache.",
        epilog="For a guaranteed full refresh, pass both flags together.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the raw dataset from Kaggle even if data/arxiv_ai.csv already exists.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-embed the dataset even if a cached embeddings/index exists in artifacts/.",
    )
    args = parser.parse_args()

    if args.force_download:
        fetch_raw_dataset(force_download=True)

    index = build_or_load_index(force_rebuild=args.force_rebuild)
    print(f"Index ready: {len(index.df)} papers, embeddings shape {index.embeddings.shape}")


if __name__ == "__main__":
    main()
