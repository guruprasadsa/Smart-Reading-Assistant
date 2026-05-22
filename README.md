#  Smart Reading Assistant

> Upload any document. Ask questions. Get cited answers with confidence indicators.

## Problem Solved

Researchers and students waste hours manually reading through PDFs and documents
to find specific information. This system lets you upload any document and ask
natural language questions — returning cited answers with source references and
confidence indicators (HIGH / PARTIAL / NOT FOUND), so you always know how
reliable each answer is.

## Architecture

```
[Upload PDF / DOCX / TXT]
        ↓
[Document Parser — PyMuPDF / python-docx / plain text]
        ↓
[RecursiveCharacterTextSplitter — 512 chars, 50 overlap]
        ↓
[Google gemini-embedding-2 → ChromaDB (persistent on disk)]
        ↓
[MMR Retrieval — k=5, fetch_k=20, lambda=0.7]
        ↓
[Partial Answer Detection — similarity score thresholding]
        ↓
[Gemini 2.5 Flash (temp=0.2) → Answer + Citations]
        ↓
[Flask REST API → Web UI]
```

## Key Components

- **Document Ingestion** (`ingestion.py`): Multi-format parsing (PDF, DOCX, TXT)
  with RecursiveCharacterTextSplitter preserving semantic boundaries
- **Vector Store** (`retrieval.py`): ChromaDB with persistent storage —
  documents survive server restarts
- **MMR Retrieval**: Maximal Marginal Relevance avoids returning redundant chunks
  from the same paragraph
- **Partial Answer Detection**: Similarity score thresholding switches the LLM
  prompt to hedged mode when confidence is low — no silent hallucinations
- **Existing Features**: HuggingFace BART summarization + spaCy key phrase
  extraction (no API key required for these)

## Engineering Challenge

Naive RAG hallucinates when no chunk fully answers a question, or returns
irrelevant text with false confidence. This system checks similarity scores
before generation: if the best retrieved chunk scores below 0.75, the prompt
switches to a hedged mode that explicitly tells the LLM to state what
information is missing rather than speculate.

## Setup

### Prerequisites
- Python 3.9+
- Google API key — get one free at https://aistudio.google.com/

### Installation

```bash
git clone https://github.com/guruprasadsa/Smart-Reading-Assistant.git
cd Smart-Reading-Assistant
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
# Open .env and add your GOOGLE_API_KEY
python app.py
# Visit http://localhost:5000
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| GOOGLE_API_KEY |  Yes | — | Google Gemini API key |
| FLASK_DEBUG | No | True | Enable Flask debug mode |
| CHROMA_PERSIST_DIR | No | ./chroma_db | ChromaDB storage path |
| UPLOAD_FOLDER | No | ./uploads | Uploaded files path |
| MAX_FILE_SIZE_MB | No | 10 | Max upload size |

## Tech Stack

Python · Flask · LangChain · ChromaDB · Google Gemini · spaCy · PyMuPDF · HuggingFace BART

## Demo

[Add Loom video link here after recording]
