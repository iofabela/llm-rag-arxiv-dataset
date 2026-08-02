"""Dataset loading: fetch the arxiv AI papers dataset via kagglehub, drop the
discarded columns, and build the text field the RAG index is built from.
"""

import os
import re

import pandas as pd

from .config import settings

# Columns explicitly flagged for removal (comment, doi).
DISCARD_COLUMNS = ["comment", "doi"]

_AUTHOR_PATTERN = re.compile(r"Author\('([^']*)'\)")


def _raw_csv_path() -> str:
    return os.path.join(settings.data_dir, settings.raw_csv_filename)


def fetch_raw_dataset(force_download: bool = False) -> pd.DataFrame:
    """Load the raw dataset.

    Reads the local CSV cache under data/ when present; otherwise downloads it
    from Kaggle via kagglehub and caches it for next time.
    """
    path = _raw_csv_path()
    if not force_download and os.path.exists(path):
        return pd.read_csv(path)

    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        settings.kaggle_dataset,
        settings.kaggle_file_path,
    )

    os.makedirs(settings.data_dir, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean_authors(raw_authors) -> str:
    """"[arxiv.Result.Author('A'), arxiv.Result.Author('B')]" -> "A, B"."""
    if not isinstance(raw_authors, str):
        return ""
    names = _AUTHOR_PATTERN.findall(raw_authors)
    return ", ".join(names) if names else raw_authors


def build_index_text(row: pd.Series) -> str:
    return f"Title: {row['title']}\nAuthors: {row['authors']}\nSummary: {row['summary']}"


def load_processed_dataset(force_download: bool = False) -> pd.DataFrame:
    """Return the dataset with discard columns dropped, authors cleaned, and
    an `index_text` column (title + authors + summary) ready for indexing.
    """
    df = fetch_raw_dataset(force_download=force_download)

    keep_columns = [c for c in df.columns if c not in DISCARD_COLUMNS]
    df = df[keep_columns].copy()

    df["authors"] = df["authors"].apply(clean_authors)
    df = df.dropna(subset=["title", "summary"]).reset_index(drop=True)
    df["index_text"] = df.apply(build_index_text, axis=1)
    return df
