# app.py
from flask import Flask, render_template, request
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

app = Flask(__name__)

model_name = 'deepset/roberta-base-squad2'
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    context = request.form['context']
    question = request.form['question']

    inputs = tokenizer(question, context, return_tensors="pt")
    output = model(**inputs)
    answer_start_idx = torch.argmax(output.start_logits)
    answer_end_idx = torch.argmax(output.end_logits)
    answer_tokens = inputs['input_ids'][0, answer_start_idx:answer_end_idx + 1]
    answer = tokenizer.decode(answer_tokens)

    return render_template('answer.html', question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
