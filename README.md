# HarmonyPal
HarmonyPal is a web application that utilizes a fine-tuned GPT-2 model to provide mental health support through a chatbot interface. The project combines React for the frontend, Flask for the backend, and a custom-trained GPT-2 model for generating responses to mental health-related queries.

## Tech Stack
Frontend: React, Tailwind CSS

Backend: Flask

Machine Learning: Hugging Face Transformers (GPT-2)

Data Processing Libraries: Pandas, Scikit-learn

Framework: PyTorch

## Hugging Face Resources
Dataset: https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset

Base Model: https://huggingface.co/openai-community/gpt2

## Key Features
- Fine-tuned GPT-2 model on mental health-related queries
- Chat interface with AI-generated responses
- Flask API for handling requests and generating responses
- React frontend for user interaction

## Setup and Installation
- Clone the repository
- Set up the Python environment and install dependencies
- Run the Flask backend: python api.py
- Start the React frontend: npm start

## Disclaimer
**Important Notice:** The information provided by this chatbot is intended for general informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. 

## Limitations
- Due to computational constraints, this AI model has been trained on a limited dataset with only a few epochs.
- It may not always provide accurate or contextually appropriate responses.
- The model's knowledge is based on its training data and does not include real-time or personal information.
- The conversation may be abruptly cut off due to size limitations in the AI model. 
