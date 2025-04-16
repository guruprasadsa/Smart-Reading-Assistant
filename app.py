from flask import Flask, request, render_template
from summarizer_module import Summarizer

app = Flask(__name__)
summarizer = Summarizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary = summarizer.summarize_text(text)
    return {'summary': summary}

if __name__ == '__main__':
    app.run(debug=True)