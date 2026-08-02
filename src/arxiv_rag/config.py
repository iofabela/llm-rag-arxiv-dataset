import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    kaggle_dataset: str = os.getenv("KAGGLE_DATASET", "yasirabdaali/arxivorg-ai-research-papers-dataset")
    kaggle_file_path: str = os.getenv("KAGGLE_FILE_PATH", "arxiv_ai.csv")

    data_dir: str = os.getenv("DATA_DIR", "data")
    raw_csv_filename: str = os.getenv("RAW_CSV_FILENAME", "arxiv_ai.csv")
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")

    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    hybrid_candidate_k: int = int(os.getenv("HYBRID_CANDIDATE_K", "50"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "200"))

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    backend_url: str = os.getenv("BACKEND_URL", "http://localhost:8000")


settings = Settings()
