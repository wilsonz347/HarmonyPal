from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import traceback
import logging
import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def load_model():
    try:
        model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer
    except Exception as e:
        print({'Failed to load model': str(e)})
        return None

gpt2_model, gpt2_tokenizer = load_model()

@app.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        if gpt2_model is None or gpt2_tokenizer is None:
            logger.error("Model or tokenizer not initialized")
            return jsonify({'error': 'Model initialization failed'}), 500

        data = request.get_json()
        question = data.get('question', '')
        logger.info(f"Received question: {question}")

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        input_text = question

        inputs = gpt2_tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            return_tensors='pt',
            padding=True,
            max_length=512,
            truncation=True,
            return_attention_mask=True
        )

        with torch.no_grad():
            outputs = gpt2_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=100,
                num_return_sequences=1,
                pad_token_id=gpt2_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Generated response: {response}")

        return jsonify({'response': response}), 200

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def home():
    return jsonify({'message': 'Welcome to HarmonyPal API'}), 200

if __name__ == '__main__':
    app.run(debug=True)