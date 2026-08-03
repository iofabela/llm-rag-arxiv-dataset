"""Load the arXiv dataset into a local DuckDB database with dlt.

The source intentionally reuses the repository's CSV/Kaggle loader so the
same cleaning rules feed both the RAG index and the durable dlt destination.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import dlt
import pandas as pd

from arxiv_rag.config import settings
from arxiv_rag.data import fetch_raw_dataset, process_dataset


@dlt.resource(
    name="arxiv_papers",
    primary_key="entry_id",
    write_disposition="replace",
)
def arxiv_papers(force_download: bool = False):
    """Yield cleaned paper records for a full refresh of the destination."""
    dataframe = process_dataset(fetch_raw_dataset(force_download=force_download))
    dataframe = dataframe.astype(object).where(pd.notna(dataframe), None)
    yield from dataframe.to_dict(orient="records")


def run_pipeline(force_download: bool = False) -> dlt.LoadInfo:
    """Run the dlt pipeline and return its load metadata."""
    database_path = Path(settings.dlt_database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    Path(settings.dlt_pipelines_dir).mkdir(parents=True, exist_ok=True)

    pipeline = dlt.pipeline(
        pipeline_name="arxiv_ingestion",
        destination=dlt.destinations.duckdb(str(database_path)),
        dataset_name=settings.dlt_dataset_name,
        pipelines_dir=settings.dlt_pipelines_dir,
        progress="log",
    )
    return pipeline.run(arxiv_papers(force_download=force_download))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load arXiv papers into DuckDB with dlt.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Refresh the local CSV from Kaggle before loading it.",
    )
    args = parser.parse_args()

    info = run_pipeline(force_download=args.force_download)
    print(info)


if __name__ == "__main__":
    # Keep command-line execution explicit and easy to use from Kestra or a shell.
    os.environ.setdefault("DLT_PROGRESS", "log")
    main()
