# summarizer_module.py
"""
Text summarization using HuggingFace BART and key phrase extraction using spaCy.
Uses AutoModelForSeq2SeqLM directly (compatible with transformers v5.x where
the 'summarization' pipeline task was removed).
"""

import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy

logger = logging.getLogger(__name__)

# ─── Model Configuration ──────────────────────────────────────────────────────
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
MAX_INPUT_LENGTH = 1024
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 30

# Load summarizer model and tokenizer once at import time
logger.info("Loading summarizer model '%s'...", SUMMARIZER_MODEL)
tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
logger.info("Summarizer model loaded successfully")

# Load spaCy model for key phrase extraction
nlp = spacy.load("en_core_web_sm")


def generate_summary(text):
    """Generate a concise summary of the input text using BART.

    Args:
        text: Input text to summarize.

    Returns:
        Summary string.
    """
    # Truncate to max input length
    text = text[:MAX_INPUT_LENGTH]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=MAX_SUMMARY_LENGTH,
        min_length=MIN_SUMMARY_LENGTH,
        do_sample=False,
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def extract_key_phrases(text):
    """Extract key noun phrases from text using spaCy NLP.

    Args:
        text: Input text to analyze.

    Returns:
        List of unique key phrases (strings).
    """
    doc = nlp(text)
    phrases = list(set(chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2))
    return phrases
