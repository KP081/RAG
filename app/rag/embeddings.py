"""
Embedding Model
 
Single place to configure embeddings so swapping models (e.g. OpenAI →
a self-hosted sentence-transformers model) is a one-line change here,
not scattered throughout the codebase.
"""

from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings

@lru_cache(maxsize=1)
def get_emmbedings() -> OpenAIEmbeddings:
    """
    Return a cached OpenAI embeddings instance.
    """
    settings = get_settings()
    
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key
    )