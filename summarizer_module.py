# summarizer_module.py
"""
Text summarization using Google Gemini API and key phrase extraction using spaCy.
"""

import logging
import spacy
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Load spaCy model for key phrase extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("Spacy model not found. Ensure 'python -m spacy download en_core_web_sm' is run.")
    nlp = None

def generate_summary(text):
    """Generate a concise summary of the input text using Gemini Flash.

    Args:
        text: Input text to summarize.

    Returns:
        Summary string.
    """
    try:
        # Use gemini-2.5-flash for summarization
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=250,
            timeout=30,
            max_retries=2,
        )
        
        prompt = f"""
        Summarize the following text in a concise and clear manner. 
        Keep the summary between 30 and 150 words.
        
        TEXT TO SUMMARIZE:
        {text[:5000]} # Limit to 5000 chars for context window
        """
        
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        logger.error("Summarization failed: %s", str(e))
        return "Summary could not be generated due to an error."

def extract_key_phrases(text):
    """Extract key noun phrases from text using spaCy NLP.

    Args:
        text: Input text to analyze.

    Returns:
        List of unique key phrases (strings).
    """
    if not nlp:
        return []
    doc = nlp(text)
    phrases = list(set(chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2))
    return phrases
