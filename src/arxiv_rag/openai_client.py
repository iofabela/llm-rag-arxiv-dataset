from functools import lru_cache

from openai import OpenAI

from .config import settings


@lru_cache
def get_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=settings.openai_api_key)
