"""
retrieval.py — ChromaDB Vector Store + MMR Retrieval + Partial Answer Detection

Manages the ChromaDB vector store with Voyage AI embeddings, MMR-based retrieval,
and confidence-aware query answering using Google Gemini. The key engineering
challenge is partial answer detection: when retrieved chunks don't fully match
a question, the system honestly communicates uncertainty instead of hallucinating.

Embeddings: Voyage AI voyage-4-large (MoE architecture)
LLM:        Google Gemini (for query answering)

Exports:
    VectorStoreManager  — ChromaDB lifecycle, add/list/clear/retrieve
    RAGQueryEngine      — LLM querying with partial answer detection
"""

import os
import re
import json
import time
import uuid
import logging
from typing import Dict, List, Optional, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import Chroma

from secrets_utils import get_api_key

logger = logging.getLogger(__name__)

# ─── Embedding Configuration ──────────────────────────────────────────────────
EMBEDDING_MODEL = "voyage-4-large"  # Voyage AI MoE embedding model (state-of-the-art retrieval)

# ─── Retrieval Configuration ───────────────────────────────────────────────────
RETRIEVAL_K = 5           # Number of chunks to return (NOT 50 — would exceed context)
RETRIEVAL_FETCH_K = 20    # Candidates considered before MMR re-ranking
RETRIEVAL_LAMBDA = 0.7    # MMR balance: 1.0=pure relevance, 0.0=pure diversity

# ─── Confidence ────────────────────────────────────────────────────────────────
# Confidence is assessed by the LLM itself (not by embedding distance thresholds).
# The LLM reads the retrieved context and the question, then semantically
# determines whether the context fully, partially, or does not answer the query.
# This is model-agnostic and does not break when embedding models change.

# ─── LLM Configuration ────────────────────────────────────────────────────────
LLM_MODEL = "gemini-3.1-flash-lite"
LLM_TEMPERATURE = 0.2     # Low temperature for factual Q&A (NOT 1.0)
LLM_MAX_TOKENS = 1024




# ─── Prompt Template ──────────────────────────────────────────────────────────
# Single prompt: the LLM answers AND self-assesses confidence in one call.
RAG_PROMPT = (
    "You are a precise document assistant. Answer the question using ONLY "
    "the information in the provided context. Be specific and cite details "
    "from the context. Do not speculate beyond what the context states.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Respond in this exact JSON format and nothing else:\n"
    '{{\n'
    '  "answer": "Your detailed answer here, citing the context.",\n'
    '  "confidence": "HIGH or PARTIAL or NOT_FOUND",\n'
    '  "partial_note": null\n'
    '}}\n\n'
    "Confidence rules (apply these strictly):\n"
    "- HIGH: The context clearly and fully answers the question.\n"
    "- PARTIAL: The context contains some relevant information but does not "
    "fully answer the question. Set partial_note to a brief explanation of "
    "what information is missing.\n"
    "- NOT_FOUND: The context does not contain information relevant to the "
    "question. Set partial_note to explain that the documents do not cover "
    "this topic.\n\n"
    "Return ONLY valid JSON. No markdown fences, no extra text."
)


class VectorStoreManager:
    """Manages the ChromaDB vector store with persistent storage.

    Handles document embedding, storage, retrieval, and lifecycle management.
    Uses Voyage AI's voyage-4-large model for vector embeddings and ChromaDB
    for persistent vector storage that survives server restarts.
    """

    def __init__(self, persist_directory: str, api_key: Optional[str] = None):
        """Initialize ChromaDB with persistent storage and Voyage AI embeddings.

        Args:
            persist_directory: Path where ChromaDB stores data on disk.
            api_key: Google API key for LLM (falls back to GOOGLE_API_KEY env var).

        Raises:
            ValueError: If no Google API key or Voyage AI API key is found.
        """
        self.persist_directory = persist_directory
        self._api_key = api_key or get_api_key()

        if not self._api_key:
            raise ValueError(
                "Google API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file, "
                "or ensure the Cloud Run service account has Secret Manager Secret Accessor role."
            )

        voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_api_key:
            raise ValueError(
                "Voyage AI API key is required. Set VOYAGE_API_KEY in your .env file."
            )

        # Use Voyage AI embeddings via the official LangChain integration
        self._embeddings = VoyageAIEmbeddings(
            model=EMBEDDING_MODEL,
            voyage_api_key=voyage_api_key,
        )

        # Load existing ChromaDB or create new one
        self._vectorstore = Chroma(
            collection_name="smart_reading_assistant",
            persist_directory=self.persist_directory,
            embedding_function=self._embeddings,
        )

        logger.info(
            "VectorStoreManager initialized (persist_dir='%s', embedding='%s')",
            self.persist_directory,
            EMBEDDING_MODEL,
        )

    def add_documents(self, documents: List[Document]) -> int:
        """Add chunked documents to ChromaDB.

        Generates unique UUIDs for each chunk to avoid ID collisions.

        Args:
            documents: List of LangChain Document objects with metadata.

        Returns:
            Number of chunks actually added (int).
        """
        if not documents:
            logger.warning("No documents provided to add_documents()")
            return 0

        try:
            # Batch in small groups to respect Voyage AI free-tier rate limits
            # (3 RPM, 10K TPM). Each batch of 10 chunks ≈ 5K tokens.
            batch_size = 10
            total_added = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                batch_ids = [str(uuid.uuid4()) for _ in batch]
                self._vectorstore.add_documents(batch, ids=batch_ids)
                total_added += len(batch)
                logger.info(
                    "Added batch %d/%d (%d chunks) to vector store",
                    (i // batch_size) + 1,
                    (len(documents) + batch_size - 1) // batch_size,
                    len(batch),
                )
                # Rate-limit delay between batches (skip after last batch)
                if i + batch_size < len(documents):
                    time.sleep(21)  # 21s to stay well within 3 RPM
            logger.info("Added %d total chunks to vector store", total_added)
            return total_added
        except Exception as e:
            logger.error("Failed to add documents to vector store: %s", str(e))
            raise

    def get_document_list(self) -> List[Dict[str, Any]]:
        """Return list of unique documents in the vector store with chunk counts.

        Returns:
            [{"filename": str, "chunk_count": int}, ...] sorted by filename.
            Returns [] if collection is empty or on any error.
        """
        try:
            collection = self._vectorstore._collection
            results = collection.get(include=["metadatas"])

            if not results["metadatas"]:
                return []

            # Count chunks per unique source filename
            doc_counts: Dict[str, int] = {}
            for meta in results["metadatas"]:
                source = meta.get("source", "unknown")
                doc_counts[source] = doc_counts.get(source, 0) + 1

            return sorted(
                [
                    {"filename": name, "chunk_count": count}
                    for name, count in doc_counts.items()
                ],
                key=lambda d: d["filename"],
            )
        except Exception as e:
            logger.error("Failed to get document list: %s", str(e))
            return []

    def clear(self) -> None:
        """Delete all documents from ChromaDB collection."""
        try:
            collection = self._vectorstore._collection
            ids = collection.get()["ids"]
            if ids:
                collection.delete(ids=ids)
            logger.info("Cleared %d chunks from vector store", len(ids))
        except Exception as e:
            logger.error("Failed to clear vector store: %s", str(e))
            raise

    def get_retriever(self):
        """Return MMR retriever with tuned search parameters.

        Uses Maximal Marginal Relevance (MMR) instead of simple similarity
        search to avoid returning near-identical chunks from the same paragraph.

        Returns:
            LangChain retriever configured with MMR search.
        """
        return self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVAL_K,              # Final chunks returned
                "fetch_k": RETRIEVAL_FETCH_K,  # Candidates before MMR re-ranking
                "lambda_mult": RETRIEVAL_LAMBDA,  # Balance relevance vs diversity
            },
        )

    def has_documents(self) -> bool:
        """Check if the vector store contains any documents.

        Returns:
            True if there are documents in the store.
        """
        try:
            collection = self._vectorstore._collection
            return collection.count() > 0
        except Exception:
            return False


class RAGQueryEngine:
    """Handles LLM querying with partial answer detection.

    The key engineering challenge: naive RAG either hallucinates when no chunk
    fully answers a question, or returns irrelevant text confidently. Instead of
    relying on embedding distance thresholds (which vary across models), this
    engine asks the LLM to self-assess confidence by reading the retrieved
    context and determining whether it actually answers the question.
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        """Initialize Gemini LLM and retriever.

        Args:
            vector_store_manager: An initialized VectorStoreManager instance.
        """
        self._vsm = vector_store_manager
        api_key = vector_store_manager._api_key

        self._llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,  # 0.2 for factual Q&A (NOT 1.0)
            max_output_tokens=LLM_MAX_TOKENS,
            google_api_key=api_key,
        )

        self._retriever = vector_store_manager.get_retriever()

        logger.info(
            "RAGQueryEngine initialized (model='%s', temp=%.1f)",
            LLM_MODEL,
            LLM_TEMPERATURE,
        )

    @staticmethod
    def _parse_llm_response(raw: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response, handling markdown fences and quirks.

        Falls back to treating the raw text as a plain answer with HIGH
        confidence if JSON parsing fails — avoids crashing on LLM format errors.
        """
        text = raw.strip()

        # Strip markdown code fences if the LLM wraps its output
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            parsed = json.loads(text)
            # Validate expected keys
            answer = parsed.get("answer", text)
            confidence = parsed.get("confidence", "HIGH").upper()
            if confidence not in ("HIGH", "PARTIAL", "NOT_FOUND"):
                confidence = "HIGH"
            partial_note = parsed.get("partial_note")
            # Normalize null-like values
            if partial_note in (None, "null", "None", ""):
                partial_note = None
            return {
                "answer": answer,
                "confidence": confidence,
                "partial_note": partial_note,
            }
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "LLM did not return valid JSON — using raw text as answer. "
                "Raw response (first 200 chars): %s",
                text[:200],
            )
            return {
                "answer": text,
                "confidence": "HIGH",
                "partial_note": None,
            }

    def query(self, question: str) -> Dict[str, Any]:
        """Full RAG query with LLM-based confidence assessment.

        Pipeline:
        1. Retrieve chunks with relevance scores (for sources + logging)
        2. Early exit if no chunks found
        3. Build context from retrieved chunks
        4. Single LLM call: answer the question AND self-assess confidence
        5. Parse structured JSON response
        6. Build deduplicated sources list
        7. Return answer + confidence + sources + partial_note

        Confidence is determined by the LLM reading the context, not by
        embedding distance thresholds. This is model-agnostic.

        Args:
            question: The user's natural language question.

        Returns:
            Dict with keys: answer, confidence, sources, partial_note.
        """
        try:
            # Step 1 — Retrieve chunks with scores
            results_with_scores = (
                self._vsm._vectorstore.similarity_search_with_relevance_scores(
                    question, k=RETRIEVAL_K
                )
            )

            # Step 2 — Early exit if nothing retrieved
            if not results_with_scores:
                return {
                    "answer": "I could not find any relevant information in the uploaded documents.",
                    "confidence": "NOT_FOUND",
                    "sources": [],
                    "partial_note": (
                        "No documents have been uploaded, or the question is "
                        "unrelated to the uploaded content."
                    ),
                }

            # Log chunk scores for debugging (no thresholding decisions here)
            for i, (doc, score) in enumerate(results_with_scores):
                src = doc.metadata.get("source", "unknown")
                logger.info(
                    "  chunk %d: score=%.4f  source=%s  preview=%.80s",
                    i, score, src, doc.page_content.replace("\n", " "),
                )

            # Step 3 — Build context string
            docs = [doc for doc, _ in results_with_scores]
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            # Step 4 — Single LLM call: answer + self-assessed confidence
            prompt_text = RAG_PROMPT.format(context=context, question=question)
            response = self._llm.invoke([HumanMessage(content=prompt_text)])

            # Handle response.content being a list of dicts (newer langchain versions)
            # e.g. [{'type': 'text', 'text': '...'}, ...]
            raw_content = response.content
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    else:
                        text_parts.append(str(part))
                raw_response = " ".join(text_parts).strip()
            else:
                raw_response = raw_content.strip()

            # Step 5 — Parse structured response
            parsed = self._parse_llm_response(raw_response)
            logger.info(
                "LLM confidence assessment: %s", parsed["confidence"]
            )

            # Step 6 — Build deduplicated sources list
            sources = []
            seen = set()
            for doc, score in results_with_scores:
                filename = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", 0)
                key = (filename, page)
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "filename": filename,
                        "chunk_preview": doc.page_content[:200].strip() + "...",
                        "page": page,
                        "score": round(float(score), 3),
                    })

            # Step 7 — Return
            return {
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "sources": sources,
                "partial_note": parsed["partial_note"],
            }

        except Exception as e:
            logger.error("RAG query failed: %s", str(e))
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "confidence": "NOT_FOUND",
                "sources": [],
                "partial_note": "Query failed due to an internal error.",
            }
