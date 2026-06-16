from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM Configuration
    openai_api_key: str
    primary_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-4o-mini"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "production_api"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    # RAG Configuration
    embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = (
        "./chroma_db"  # local disk; swap for managed Chroma in prod
    )
    collection_name: str = "production_rag"
    retrieval_k: int = 4  # top-k docs to retrieve per query
    chunk_size: int = 500
    chunk_overlap: int = 50

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings — loaded once from .env, reused everywhere."""
    return Settings()
