# summarizer_module.py

from transformers import pipeline
import spacy

# Load summarizer and spaCy model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

def generate_summary(text):
    # Limit long input text for basic summarizer
    max_input = 1024
    text = text[:max_input]
    result = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return result[0]['summary_text']

def extract_key_phrases(text):
    doc = nlp(text)
    phrases = list(set(chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2))
    return phrases
