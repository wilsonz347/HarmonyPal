from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
import torch


app = Flask(__name__)

# Load BERT for intent classification
bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)

# Load GPT-2 for text generation
gpt2_model_name = "gpt2"
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)


# Define route for classifying intent using BERT
@app.route('/classify', methods=['POST'])
def classify_intent():
    data = request.json
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"error": "No input provided."}), 400

    # Tokenize and classify input
    inputs = bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()

    intents = {0: "seeking_help", 1: "expressing_emotion"}  # Example intents
    intent_label = intents.get(prediction, "unknown")

    return jsonify({"intent": intent_label})


# Define route for generating response using GPT-2
@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"error": "No input provided."}), 400

    # Tokenize input and generate response
    inputs = gpt2_tokenizer(user_input, return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs.input_ids,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})


@app.route('/')
def home():
    return "Mental Health Chatbot is running!"

if __name__ == "__main__":
    app.run(debug=True)