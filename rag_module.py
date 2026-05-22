"""
rag_module.py — RAG Module Interface

Thin wrapper that orchestrates the ingestion and retrieval modules.
Provides a simple API for the Flask app: initialize, add documents,
query, list documents, and clear.
"""

import os
import logging
import threading
from typing import Dict, List, Optional, Any

from ingestion import chunk_document
from retrieval import VectorStoreManager, RAGQueryEngine

logger = logging.getLogger(__name__)

# ─── Module-level singletons ───────────────────────────────────────────────────
_vector_store_manager: Optional[VectorStoreManager] = None
_query_engine: Optional[RAGQueryEngine] = None
_init_lock = threading.Lock()


def initialize_rag(
    persist_directory: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Initialize the RAG system with vector store and query engine.

    Creates the VectorStoreManager (ChromaDB) and RAGQueryEngine (Gemini)
    singletons. Safe to call multiple times — reinitializes if already set up.

    Args:
        persist_directory: Path for ChromaDB storage. Defaults to CHROMA_PERSIST_DIR
                          env var or './chroma_db'.
        api_key: Google API key. Defaults to GOOGLE_API_KEY env var.

    Raises:
        ValueError: If no Google API key is available.
    """
    global _vector_store_manager, _query_engine

    with _init_lock:
        if _vector_store_manager is not None and _query_engine is not None:
            return  # Already initialized by another thread

        persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

        _vector_store_manager = VectorStoreManager(
            persist_directory=persist_dir,
            api_key=api_key,
        )
        _query_engine = RAGQueryEngine(_vector_store_manager)

        logger.info("RAG system initialized (persist_dir='%s')", persist_dir)


def add_document_to_rag(filepath: str) -> Dict[str, Any]:
    """Ingest a document into the RAG pipeline.

    Full pipeline: parse document → chunk with RecursiveCharacterTextSplitter
    → embed with Google gemini-embedding-2 → store in ChromaDB.

    Args:
        filepath: Absolute path to the document file.

    Returns:
        Dict with keys: status, filename, chunks_created, message.

    Raises:
        RuntimeError: If RAG system is not initialized.
    """
    if _vector_store_manager is None:
        raise RuntimeError(
            "RAG system not initialized. Call initialize_rag() first."
        )

    filename = os.path.basename(filepath)

    try:
        # Parse and chunk the document
        chunks = chunk_document(filepath)

        if not chunks:
            return {
                "status": "warning",
                "filename": filename,
                "chunks_created": 0,
                "message": f"No text content found in '{filename}'.",
            }

        # Add chunks to vector store
        count = _vector_store_manager.add_documents(chunks)

        logger.info("Ingested '%s': %d chunks added to vector store", filename, count)

        return {
            "status": "success",
            "filename": filename,
            "chunks_created": count,
            "message": f"Successfully processed '{filename}' into {count} chunks.",
        }

    except ValueError as e:
        logger.warning("Document ingestion warning for '%s': %s", filename, str(e))
        return {
            "status": "error",
            "filename": filename,
            "chunks_created": 0,
            "message": str(e),
        }
    except Exception as e:
        logger.error("Document ingestion failed for '%s': %s", filename, str(e))
        return {
            "status": "error",
            "filename": filename,
            "chunks_created": 0,
            "message": f"Failed to process '{filename}': {str(e)}",
        }


def query_rag(question: str) -> Dict[str, Any]:
    """Query the RAG system with partial answer detection.

    Args:
        question: The user's natural language question.

    Returns:
        Dict with keys: answer, confidence, sources, partial_note.

    Raises:
        RuntimeError: If RAG system is not initialized.
    """
    if _query_engine is None:
        raise RuntimeError(
            "RAG system not initialized. Call initialize_rag() first."
        )

    return _query_engine.query(question)


def get_loaded_documents() -> List[Dict[str, Any]]:
    """Get list of all documents currently in the vector store.

    Returns:
        List of dicts: [{filename, chunk_count}, ...].
    """
    if _vector_store_manager is None:
        return []

    return _vector_store_manager.get_document_list()


def clear_rag() -> Dict[str, str]:
    """Clear all documents from the vector store.

    Returns:
        Dict with status and message.
    """
    if _vector_store_manager is None:
        return {"status": "warning", "message": "RAG system not initialized."}

    try:
        _vector_store_manager.clear()
        logger.info("RAG system cleared — all documents removed")
        return {"status": "success", "message": "All documents cleared."}
    except Exception as e:
        logger.error("Failed to clear RAG system: %s", str(e))
        return {"status": "error", "message": f"Failed to clear: {str(e)}"}