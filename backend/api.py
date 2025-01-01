from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
CORS(app)

def load_model():
    try:
        model = GPT2LMHeadModel.from_pretrained('./fine_tuned_mental_health_model')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer
    except Exception as e:
        print({'Failed to load model': str(e)})
        return None

tuned_model, gpt_tokenizer = load_model()

@app.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        input_text = f"Question: {question} Answer:"
        inputs = gpt_tokenizer.encode(input_text, return_tensors='pt')

        # Generate a response
        generated_ids = tuned_model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.7)
        response = gpt_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract the answer
        answer = response.split("Answer:", 1)[1].strip() if "Answer:" in response else response

        return jsonify({'response': answer}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)