# app.py

from flask import Flask, request, jsonify, render_template
from summarizer_module import generate_summary, extract_key_phrases

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Text is empty!"}), 400

    summary = generate_summary(text)
    key_phrases = extract_key_phrases(text)

    return jsonify({
        "summary": summary,
        "key_phrases": key_phrases
    })

if __name__ == "__main__":
    app.run(debug=True)
