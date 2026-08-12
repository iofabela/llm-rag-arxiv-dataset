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


def fetch_dlt_dataset() -> pd.DataFrame | None:
    """Read the dlt-managed table when a completed ingestion exists.

    Returning ``None`` for a missing table keeps local development compatible
    with the original CSV-only setup. A present but invalid database is
    allowed to raise so ingestion or storage failures are visible.
    """
    if not os.path.exists(settings.dlt_database_path):
        return None

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - dependency is project-managed
        raise RuntimeError("duckdb is required to read the dlt dataset") from exc

    def quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    connection = duckdb.connect(settings.dlt_database_path, read_only=True)
    try:
        table_exists = connection.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            """,
            [settings.dlt_dataset_name, settings.dlt_resource_name],
        ).fetchone()
        if table_exists is None:
            return None

        table = f"{quote_identifier(settings.dlt_dataset_name)}.{quote_identifier(settings.dlt_resource_name)}"
        return connection.execute(f"SELECT * FROM {table}").df()
    finally:
        connection.close()


def clean_authors(raw_authors) -> str:
    """"[arxiv.Result.Author('A'), arxiv.Result.Author('B')]" -> "A, B"."""
    if not isinstance(raw_authors, str):
        return ""
    names = _AUTHOR_PATTERN.findall(raw_authors)
    return ", ".join(names) if names else raw_authors


def build_index_text(row: pd.Series) -> str:
    return f"Title: {row['title']}\nAuthors: {row['authors']}\nSummary: {row['summary']}"


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the transformations shared by CSV and dlt ingestion paths."""
    keep_columns = [c for c in df.columns if c not in DISCARD_COLUMNS]
    df = df[keep_columns].copy()

    df["authors"] = df["authors"].apply(clean_authors)
    df = df.dropna(subset=["title", "summary"]).reset_index(drop=True)
    df["index_text"] = df.apply(build_index_text, axis=1)
    return df


def load_processed_dataset(force_download: bool = False) -> pd.DataFrame:
    """Return the latest dlt dataset, falling back to the Kaggle CSV cache."""
    # An explicit CSV refresh is useful when rebuilding the index independently
    # of the scheduled dlt flow.
    df = None if force_download else fetch_dlt_dataset()
    if df is None:
        df = fetch_raw_dataset(force_download=force_download)
    return process_dataset(df)
