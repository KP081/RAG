"""
Vector Store
 
Thin wrapper around ChromaDB that returns LangChain-compatible
retriever objects. All callers get retrievers — never raw Chroma
clients — so the underlying store can be swapped (Pinecone, Qdrant,
pgvector) without touching app/agent.py.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import get_settings
from app.rag.embeddings import get_emmbedings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Return (or create) the persistent Chroma collection.
    """
    settings = get_settings()
    
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_emmbedings(),
        persist_directory=settings.chroma_persist_dir
    )
    
def get_retrievar(k: int | None = None) -> VectorStoreRetriever:
    """
    Return a retriever over the vector store.
 
    Args:
        k: Number of documents to retrieve. Defaults to settings.retrieval_k.
 
    Production notes:
        - For MMR (Maximal Marginal Relevance) - reduces repetition in
          retrieved docs - use search_type="mmr".
        - For hybrid search (dense + BM25) plug in a different retriever
          class here without touching the agent.
    """
    
    settings = get_settings()
    
    k = k or settings.retrieval_k
    
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )