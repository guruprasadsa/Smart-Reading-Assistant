"""
app.py — Flask Web Server for Smart Reading Assistant

Routes only — all business logic is delegated to:
  - summarizer_module.py: text summarization + key phrase extraction
  - rag_module.py: document ingestion, RAG retrieval, partial answer detection

Serves a web UI with three tabs: RAG Q&A, Summarize, and About.
"""

import os
import logging
import shutil

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# ─── Environment & Logging ─────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Initialize RAG System ────────────────────────────────────────────────────
rag_initialized = False

try:
    from rag_module import (
        initialize_rag,
        add_document_to_rag,
        query_rag,
        get_loaded_documents,
        clear_rag,
    )

    initialize_rag()
    rag_initialized = True
    logger.info("RAG system initialized successfully")
except ValueError as e:
    logger.warning(
        "RAG system not initialized (missing API key): %s. "
        "RAG features will be disabled. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env file or Cloud Secrets.",
        str(e),
    )
except Exception as e:
    logger.warning(
        "RAG system initialization failed: %s. RAG features will be disabled.",
        str(e),
    )

# ─── Initialize Summarizer ────────────────────────────────────────────────────
summarizer_initialized = False

try:
    from summarizer_module import generate_summary, extract_key_phrases

    summarizer_initialized = True
    logger.info("Summarizer module loaded successfully")
except Exception as e:
    logger.warning("Summarizer module failed to load: %s", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════


@app.route("/")
def home():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze text: generate summary and extract key phrases.

    This preserves the existing summarization functionality.

    Request:
        Form data with 'text' field.

    Response:
        JSON: {summary, key_phrases}
    """
    if not summarizer_initialized:
        return jsonify({
            "error": "Summarizer module is not available. Check server logs."
        }), 503

    text = request.form.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        summary = generate_summary(text)
        key_phrases = extract_key_phrases(text)

        logger.info(
            "Analyzed text: %d chars → summary=%d chars, %d key phrases",
            len(text),
            len(summary),
            len(key_phrases),
        )

        return jsonify({
            "summary": summary,
            "key_phrases": key_phrases,
        })

    except Exception as e:
        logger.error("Analysis failed: %s", str(e))
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """Upload a document for RAG ingestion.

    Accepts PDF, DOCX, or TXT files. Processes through the ingestion pipeline:
    parse → chunk (RecursiveCharacterTextSplitter) → embed → store in ChromaDB.

    Request:
        Multipart form data with 'file' field.

    Response:
        JSON: {status, filename, chunks_created, message}
    """
    if not rag_initialized:
        return jsonify({
            "error": "RAG system is not initialized. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env file or Cloud Secrets."
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # Validate file extension
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Save to uploads folder
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(filepath)
        logger.info("Saved uploaded file: %s (%d bytes)", filename, os.path.getsize(filepath))
    except Exception as e:
        logger.error("Failed to save file '%s': %s", filename, str(e))
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    # Process through RAG ingestion pipeline
    try:
        result = add_document_to_rag(filepath)

        if result["status"] == "error":
            return jsonify(result), 422

        return jsonify(result)

    except Exception as e:
        logger.error("Ingestion failed for '%s': %s", filename, str(e))
        return jsonify({
            "error": f"Failed to process document: {str(e)}"
        }), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Ask a question against uploaded documents.

    Uses the RAG pipeline with partial answer detection:
    retrieve (MMR, k=5) → score confidence → select prompt → generate answer.

    Request:
        JSON: {question: "string"}

    Response:
        JSON: {answer, confidence, sources: [{filename, chunk_preview, page}], partial_note}
    """
    if not rag_initialized:
        return jsonify({
            "error": "RAG system is not initialized. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env file or Cloud Secrets."
        }), 503

    data = request.get_json(silent=True)

    if not data or not data.get("question", "").strip():
        return jsonify({"error": "No question provided."}), 400

    question = data["question"].strip()

    try:
        result = query_rag(question)
        logger.info(
            "Answered question (confidence=%s): '%s'",
            result.get("confidence", "?"),
            question[:80],
        )
        return jsonify(result)

    except Exception as e:
        logger.error("Query failed: %s", str(e))
        return jsonify({
            "error": f"Failed to answer question: {str(e)}"
        }), 500


@app.route("/documents", methods=["GET"])
def documents():
    """List all documents currently in the vector store.

    Response:
        JSON: {documents: [{filename, chunk_count}, ...]}
    """
    if not rag_initialized:
        return jsonify({"documents": [], "message": "RAG system not initialized."})

    try:
        doc_list = get_loaded_documents()
        return jsonify({"documents": doc_list})
    except Exception as e:
        logger.error("Failed to list documents: %s", str(e))
        return jsonify({"error": f"Failed to list documents: {str(e)}"}), 500


@app.route("/documents", methods=["DELETE"])
def delete_documents():
    """Clear all documents from the vector store and uploads folder.

    Response:
        JSON: {status, message}
    """
    if not rag_initialized:
        return jsonify({"status": "warning", "message": "RAG system not initialized."})

    try:
        # Clear vector store
        result = clear_rag()

        # Clear uploaded files
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            logger.info("Cleared upload folder: %s", UPLOAD_FOLDER)

        return jsonify(result)

    except Exception as e:
        logger.error("Failed to clear documents: %s", str(e))
        return jsonify({"error": f"Failed to clear documents: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Response:
        JSON: {status, rag_available, summarizer_available}
    """
    return jsonify({
        "status": "ok",
        "rag_available": rag_initialized,
        "summarizer_available": summarizer_initialized,
    })


# ─── Error Handlers ───────────────────────────────────────────────────────────


@app.errorhandler(413)
def file_too_large(e):
    """Handle file size exceeding MAX_FILE_SIZE_MB."""
    return jsonify({
        "error": f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle unexpected server errors."""
    return jsonify({"error": "Internal server error. Check server logs."}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    logger.info("Starting Smart Reading Assistant (debug=%s)", debug_mode)
    port = int(os.getenv("PORT", 5000))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
