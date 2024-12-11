from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForSequenceClassification
import torch
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate

# Load GPT-2
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load BERT for sentiment analysis
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Sentiment Analysis (BERT)
def analyze_sentiment(input_text):
    inputs = bert_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()
    return "positive" if sentiment == 1 else "negative"

# Text Generation (GPT-2)
def generate_response(context):
    inputs = gpt2_tokenizer.encode(context, return_tensors="pt", max_length=512, truncation=True)
    outputs = gpt2_model.generate(inputs, max_length=150, num_return_sequences=1)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def chatbot_response(input_text):
    sentiment = analyze_sentiment(input_text)
    context = f"The user seems {sentiment}. Here's an empathetic response: {input_text}"
    response = generate_response(context)
    return response

