"""
Document Ingestion Pipeline

Loads documents from a directory, chunks them, embeds them, and stores
them in ChromaDB.  Run this ONCE before starting the API (and again
whenever your knowledge base is updated).

Usage:
    python -m app.rag.ingestion --source ./docs
    python -m app.rag.ingestion --source ./docs --clear   # wipe + re-index

Supported file types: .txt, .md
(Add PyPDF loader trivially for .pdf support — see comments below.)
"""

import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.monitoring import get_logger
from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import get_vectorstore

logger = get_logger("ingestion")


def ingest(source_dir: str, clear: bool = False) -> int:
    """
    Load, chunk, embed, and store documents.

    Args:
        source_dir: Path to the directory containing source documents.
        clear: If True, wipe the existing collection before indexing.

    Returns:
        Number of chunks stored.
    """
    settings = get_settings()
    source = Path(source_dir)

    if not source.exists():
        logger.error(f"Source directory not found: {source}")
        sys.exit(1)

    # Load documents
    logger.info(f"Loading documents from {source}...")
    loader = DirectoryLoader(
        str(source),
        glob="**/*.{txt,md}",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )

    # PDF loader
    pdf_loader = PyPDFDirectoryLoader(str(source))
    documents = loader.load() + pdf_loader.load()

    if not documents:
        logger.warning(f"No .txt or .md files found in {source}")
        return 0

    logger.info(f"Loaded {len(documents)} documents.")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks.")

    # Store
    vs = get_vectorstore()

    if clear:
        logger.info("Clearing existing collection...")
        vs.delete_collection()
        # Re-create after deletion
        from app.rag.vectorstore import get_vectorstore as _vs
        get_vectorstore.cache_clear()     # bust lru_cache
        vs = _vs()

    vs.add_documents(chunks)
    logger.info(f"Stored {len(chunks)} chunks in '{settings.collection_name}'.")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--source", default="./docs", help="Directory of source documents.")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the collection before ingesting (full re-index).",
    )
    args = parser.parse_args()

    count = ingest(args.source, clear=args.clear)
    print(f"\n✓ Ingestion complete. {count} chunks indexed.")