"""
retrieval.py — ChromaDB Vector Store + MMR Retrieval + Partial Answer Detection

Manages the ChromaDB vector store with Google embeddings, MMR-based retrieval,
and confidence-aware query answering using Google Gemini. The key engineering
challenge is partial answer detection: when retrieved chunks don't fully match
a question, the system honestly communicates uncertainty instead of hallucinating.

Exports:
    VectorStoreManager  — ChromaDB lifecycle, add/list/clear/retrieve
    RAGQueryEngine      — LLM querying with partial answer detection
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from google import genai
from google.genai import types

from secrets_utils import get_api_key

logger = logging.getLogger(__name__)

# ─── Embedding Configuration ──────────────────────────────────────────────────
EMBEDDING_MODEL = "models/gemini-embedding-2"  # Google embedding model (NOT a chat model)

# ─── Retrieval Configuration ───────────────────────────────────────────────────
RETRIEVAL_K = 5           # Number of chunks to return (NOT 50 — would exceed context)
RETRIEVAL_FETCH_K = 20    # Candidates considered before MMR re-ranking
RETRIEVAL_LAMBDA = 0.7    # MMR balance: 1.0=pure relevance, 0.0=pure diversity

# ─── Confidence Thresholds ─────────────────────────────────────────────────────
CONFIDENCE_HIGH_THRESHOLD = 0.75
CONFIDENCE_PARTIAL_THRESHOLD = 0.40

# ─── LLM Configuration ────────────────────────────────────────────────────────
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2     # Low temperature for factual Q&A (NOT 1.0)
LLM_MAX_TOKENS = 1024


# ─── Custom Embedding Wrapper ────────────────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    """Custom embedding wrapper using the google-genai SDK directly.

    This bypasses the LangChain GoogleGenerativeAIEmbeddings wrapper to fix
    batch embedding bugs and ensure reliable Content construction.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in batches of 100 (Google API limit)."""
        batch_size = 100
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            contents = [
                types.Content(parts=[types.Part.from_text(text=text)])
                for text in batch
            ]
            response = self.client.models.embed_content(
                model=self.model,
                contents=contents,
            )
            embeddings.extend([list(emb.values) for emb in response.embeddings])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return list(response.embeddings[0].values)


# ─── Prompt Templates ─────────────────────────────────────────────────────────
PROMPT_HIGH_CONFIDENCE = (
    "You are a precise document assistant. Answer the question using "
    "ONLY the information in the provided context. Be specific and "
    "cite details from the context. If the context doesn't fully "
    "address the question, say so clearly.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

PROMPT_PARTIAL_CONFIDENCE = (
    "You are a precise document assistant. The retrieved documents "
    "may not fully address this question. Answer ONLY what the "
    "documents explicitly support. Clearly state what information "
    "is missing or uncertain. Do not speculate beyond the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer only what the context supports, and note any gaps:"
)


class VectorStoreManager:
    """Manages the ChromaDB vector store with persistent storage.

    Handles document embedding, storage, retrieval, and lifecycle management.
    Uses Google's embedding-001 model for vector embeddings and ChromaDB
    for persistent vector storage that survives server restarts.
    """

    def __init__(self, persist_directory: str, api_key: Optional[str] = None):
        """Initialize ChromaDB with persistent storage and Google embeddings.

        Args:
            persist_directory: Path where ChromaDB stores data on disk.
            api_key: Google API key (falls back to GOOGLE_API_KEY env var).

        Raises:
            ValueError: If no Google API key is found.
        """
        self.persist_directory = persist_directory
        self._api_key = api_key or get_api_key()

        if not self._api_key:
            raise ValueError(
                "Google API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file, "
                "or ensure the Cloud Run service account has Secret Manager Secret Accessor role."
            )

        # Use custom embedding wrapper for reliability
        self._embeddings = GeminiEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=self._api_key,
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
            ids = [str(uuid.uuid4()) for _ in documents]
            self._vectorstore.add_documents(documents, ids=ids)
            logger.info("Added %d chunks to vector store", len(documents))
            return len(documents)
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
    fully answers a question, or returns irrelevant text confidently. This engine
    checks relevance scores BEFORE generating an answer and switches between
    confident and hedged prompts based on retrieval quality.
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
        self._confidence_threshold = CONFIDENCE_HIGH_THRESHOLD

        logger.info(
            "RAGQueryEngine initialized (model='%s', temp=%.1f)",
            LLM_MODEL,
            LLM_TEMPERATURE,
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Full RAG query with partial answer detection.

        Pipeline:
        1. Retrieve chunks with relevance scores
        2. Check confidence via similarity score thresholds
        3. Build context from retrieved chunks
        4. Select prompt based on confidence level
        5. Call LLM with the selected prompt
        6. Build deduplicated sources list
        7. Return answer + confidence + sources + partial_note

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

            # Step 2 — Check confidence
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

            max_score = max(score for _, score in results_with_scores)
            logger.info("Max relevance score: %.3f", max_score)

            if max_score >= CONFIDENCE_HIGH_THRESHOLD:
                confidence = "HIGH"
                partial_note = None
            elif max_score >= CONFIDENCE_PARTIAL_THRESHOLD:
                confidence = "PARTIAL"
                partial_note = (
                    "The documents partially address this question. "
                    "The answer may be incomplete."
                )
            else:
                confidence = "NOT_FOUND"
                partial_note = (
                    "The uploaded documents do not appear to contain "
                    "relevant information for this question."
                )

            # Step 3 — Build context string
            docs = [doc for doc, _ in results_with_scores]
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            # Step 4 — Select prompt based on confidence
            if confidence == "HIGH":
                prompt_text = PROMPT_HIGH_CONFIDENCE.format(
                    context=context, question=question
                )
            else:
                prompt_text = PROMPT_PARTIAL_CONFIDENCE.format(
                    context=context, question=question
                )

            # Step 5 — Call LLM
            response = self._llm.invoke([HumanMessage(content=prompt_text)])
            answer = response.content.strip()

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

            # Step 8 — Return
            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "partial_note": partial_note,
            }

        # Step 7 — Error handling
        except Exception as e:
            logger.error("RAG query failed: %s", str(e))
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "confidence": "NOT_FOUND",
                "sources": [],
                "partial_note": "Query failed due to an internal error.",
            }
