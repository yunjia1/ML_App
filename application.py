from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json
import time

# Initialize Flask app
application = Flask(__name__)

# Load model and vectorizer
loaded_model = None
vectorizer = None

def load_model():
    global loaded_model, vectorizer
    try:
        with open('basic_classifier.pkl', 'rb') as fid:
            loaded_model = pickle.load(fid)
        with open('count_vectorizer.pkl', 'rb') as fid:
            vectorizer = pickle.load(fid)
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

# Load models when the module is imported
load_model()

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

@application.route("/predict", methods=['POST'])
def predict():
    start_time = time.time()
    
    # Check if models are loaded
    if loaded_model is None or vectorizer is None:
        if not load_model():  # Try to load models if they're not loaded
            return jsonify({'error': 'Models not initialized properly'}), 500
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Transform the text and make prediction
        text_vectorized = vectorizer.transform([text])
        prediction = loaded_model.predict(text_vectorized)[0]
        
        response = {
            'text': text,
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'processing_time': time.time() - start_time
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    application.run(port=5000, debug=True)