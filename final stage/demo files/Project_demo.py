from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("./saved_bert_model")
tokenizer = BertTokenizerFast.from_pretrained("./saved_bert_model")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods = ['POST'])
def classify_email():
    email_text = request.form.get('email_text', '')
    if not email_text:
        return jsonify({"error": "No email text provided"}), 400
    
    inputs = tokenizer(email_text, return_tensors = "pt", truncation = True, padding = True, max_length = 32)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim = 1)
        predicted_class = torch.argmax(probabilities, dim = 1).item()

    label = "Spam" if predicted_class == 1 else "Ham"
    confidence = probabilities[0][predicted_class].item()
    return render_template('index.html', prediction = label, confidence = f"{confidence:.2f}")

if __name__ == "__main__":
    app.run(debug = True)
