# 📄 Smart Reading Assistant

A document question-answering system that ingests PDF, DOCX, and TXT files and returns cited answers with confidence indicators using retrieval-augmented generation.
Try it live at [smart-reading-assistant.run.app](https://smart-reading-assistant-226917960220.asia-south1.run.app).

## Problem

Researchers, students, and analysts routinely spend hours skimming through long documents to locate specific information. Search tools find keywords but not answers. This project provides a natural language interface over uploaded documents — returning answers grounded in source text, with explicit confidence levels (HIGH, PARTIAL, NOT_FOUND) so the user always knows when the system is uncertain rather than guessing.

## Architecture

```
[Upload PDF / DOCX / TXT]
        ↓
[Document Parser — PyMuPDF / python-docx / plain text]
        ↓
[RecursiveCharacterTextSplitter — 512 tokens, 50 overlap]
        ↓
[Voyage AI voyage-4-large → ChromaDB (persistent)]
        ↓
[MMR Retrieval — k=5, fetch_k=20, λ=0.7]
        ↓
[Confidence Assessment — LLM semantic analysis]
        ↓
[Gemini 3.1 Flash Lite (temp=0.2) → Answer + Citations]
        ↓
[Flask REST API → Web UI]
```

RecursiveCharacterTextSplitter was chosen over fixed-size chunking because it respects paragraph and sentence boundaries, producing chunks that preserve semantic coherence. The 512-token chunk size with 50-token overlap balances retrieval granularity against context completeness — small enough for precise matching, with enough overlap to avoid splitting mid-sentence at boundaries. MMR (Maximal Marginal Relevance) replaces simple top-k retrieval to diversify the retrieved chunks; without it, the top 5 results often come from the same paragraph, giving the LLM redundant context. Confidence assessment gates generation quality — the LLM self-evaluates whether the context is sufficient to answer the question, switching to a hedged response (PARTIAL) or declaring NOT_FOUND rather than producing a confident-sounding hallucination.

## Key Components

| File | Responsibility | Notes |
|---|---|---|
| ingestion.py | Document parsing and chunking | Supports PDF, DOCX, TXT |
| retrieval.py | Vector store management and query engine | ChromaDB + MMR + partial answer detection |
| rag_module.py | Orchestration layer | Thin wrapper, delegates to ingestion and retrieval |
| app.py | Flask REST API | Routes only — no business logic |
| summarizer_module.py | Text summarization and key phrase extraction | Google Gemini 2.5 Flash + spaCy |

## API Reference

```
GET  /
```
Serves the web UI.

---

```
GET  /health
```
Returns system status.
```json
{"status": "ok", "rag_available": true, "summarizer_available": true}
```

---

```
POST /upload
```
Upload a document for RAG ingestion.

Body: `multipart/form-data` — field: `file` (.pdf, .docx, .txt, max 10MB)
```json
{"status": "success", "filename": "paper.pdf", "chunks_created": 42, "message": "Document processed successfully"}
```

---

```
POST /ask
```
Query the RAG pipeline.

Body: `{"question": "string"}`
```json
{
  "answer": "...",
  "confidence": "HIGH|PARTIAL|NOT_FOUND",
  "sources": [{"filename": "paper.pdf", "chunk_preview": "...", "page": 3}],
  "partial_note": null
}
```

---

```
GET  /documents
```
List all documents currently in the vector store.
```json
{"documents": [{"filename": "paper.pdf", "chunk_count": 42}]}
```

---

```
DELETE /documents
```
Remove all documents from the vector store.
```json
{"status": "success", "message": "All documents cleared"}
```

---

```
POST /analyze
```
Summarize text and extract key phrases (no API key required).

Body: `form-data` — field: `text`
```json
{"summary": "...", "key_phrases": ["retrieval augmented generation", "vector store"]}
```

## Engineering Notes

The central challenge in any RAG system is what happens when retrieved context does not actually contain the answer. Naive implementations pass low-relevance chunks to the LLM regardless of match quality, producing responses that sound authoritative but are fabricated. This system addresses that by asking the LLM to self-assess confidence. The LLM reads the retrieved context and the question, then semantically determines whether the context fully (HIGH), partially (PARTIAL), or does not (NOT_FOUND) answer the query, honest about what is missing without relying on embedding distance thresholds that vary across models.

Simple top-k retrieval has a second, subtler failure mode: when a document discusses the same concept across multiple paragraphs, the top 5 embeddings often come from adjacent or overlapping chunks in the same section. The LLM receives five variations of the same context and misses relevant information elsewhere in the document. MMR re-ranks candidates using a diversity penalty (λ=0.7), selecting chunks that are individually relevant to the query but dissimilar to each other. This gives the model broader coverage of the document and produces more complete answers, particularly for questions that span multiple sections.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9+, Flask |
| LLM | Google Gemini 3.1 Flash Lite |
| Embeddings | Voyage AI voyage-4-large |
| Vector Store | ChromaDB (persistent) |
| Retrieval | LangChain MMR |
| Document Parsing | PyMuPDF, python-docx |
| Summarization | Google Gemini 2.5 Flash |
| NLP | spaCy (en_core_web_sm) |
| Deployment | Google Cloud Run (asia-south1) |
| Containerization | Docker |

## Deployment

Live instance: [smart-reading-assistant.run.app](https://smart-reading-assistant-226917960220.asia-south1.run.app)

The application is containerized with Docker and deployed to Google Cloud Run in the asia-south1 region. Cloud Run handles scaling, HTTPS termination, and container lifecycle automatically. The API itself is stateless — ChromaDB manages its own persistence to disk, so the vector store survives container restarts as long as the storage volume is retained.

## Local Setup

### Prerequisites
- Python 3.9+
- Google API key ([aistudio.google.com](https://aistudio.google.com/))

### Installation

```bash
git clone https://github.com/guruprasadsa/Smart-Reading-Assistant.git
cd Smart-Reading-Assistant
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
python app.py
# Application runs at http://localhost:5000
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| GOOGLE_API_KEY | Yes | — | Google Gemini API key |
| FLASK_DEBUG | No | True | Flask debug mode |
| CHROMA_PERSIST_DIR | No | ./chroma_db | ChromaDB storage path |
| UPLOAD_FOLDER | No | ./uploads | Uploaded files directory |
| MAX_FILE_SIZE_MB | No | 10 | Upload size limit |

## Demo

Demo video: [to be added]

## License

MIT
