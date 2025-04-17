from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import json

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample document store
doc_store = [
    "Artificial Intelligence is transforming industries by automating tasks and extracting insights from data.",
    "Machine learning models can be used to predict outcomes based on large datasets.",
    "Generative AI models can create realistic text, images, and even music using training data."
]
doc_embeddings = embed_model.encode(doc_store)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))


def summarize_text(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']


def extract_key_phrases(text, num=10):
    words = word_tokenize(text)
    words = [w.lower() for w in words if w.isalnum()]
    stop_words = set(stopwords.words('english'))
    keywords = [w for w in words if w not in stop_words]
    freq = nltk.FreqDist(keywords)
    return [w for w, _ in freq.most_common(num)]


def retrieve_similar_documents(query, k=2):
    q_embedding = embed_model.encode([query])
    D, I = index.search(np.array(q_embedding), k)
    return [doc_store[i] for i in I[0]]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    summary = summarize_text(text)
    key_phrases = extract_key_phrases(text)
    related_docs = retrieve_similar_documents(text)
    
    json_output = json.dumps({
        "summary": summary,
        "keyPhrases": key_phrases,
        "additionalContext": related_docs
    }, indent=2)

    return jsonify({
        "summary": summary,
        "key_phrases": key_phrases,
        "related_docs": related_docs,
        "structured_output": json_output
    })


if __name__ == '__main__':
    app.run(debug=True)
