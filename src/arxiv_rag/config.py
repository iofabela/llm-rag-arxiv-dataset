import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

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

    es_url: str = os.getenv("ES_URL", "http://localhost:9200")
    es_api_key: str = os.getenv("ES_API_KEY", "")
    es_index_name: str = os.getenv("ES_INDEX_NAME", "arxiv_papers")

    # One of: hybrid_rerank, hybrid_only, elasticsearch_rrf (see strategies.py).
    # elasticsearch_rrf won the 200-sample comparison in
    # notebooks/retrieval_evaluation.py (composite score 0.763 vs 0.663 for
    # hybrid_rerank) -- requires Elasticsearch running (docker compose up -d).
    retrieval_strategy: str = os.getenv("RETRIEVAL_STRATEGY", "elasticsearch_rrf")

    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "200"))

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    backend_url: str = os.getenv("BACKEND_URL", "http://localhost:8000")


settings = Settings()
