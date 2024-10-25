from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json
import time
import re
# Initialize Flask app
application = Flask(__name__)

# Load model and vectorizer
loaded_model = None
vectorizer = None

def load_model():
    try:
        with open('basic_classifier.pkl', 'rb') as fid:
            loaded_model = pickle.load(fid)
        with open('count_vectorizer.pkl', 'rb') as vd:
            vectorizer = pickle.load(vd)
        print("Models loaded successfully!")
        return loaded_model, vectorizer
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

# Load models when the module is imported

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

def validate_input(text):
    """Validate the input text for basic checks"""
    # Check if text is a string
    if not isinstance(text, str):
        return False, "Text must be a string."

    # Check if text length is reasonable (e.g., between 10 and 5000 characters)
    if not (10 <= len(text) <= 5000):
        return False, "Text length must be between 10 and 5000 characters."

    # Check if text contains only valid characters (letters, numbers, punctuation)
    if not re.match(r"^[a-zA-Z0-9\s.,!?'-]+$", text):
        return False, "Text contains invalid characters."

    return True, ""

@application.route("/predict", methods=['POST'])
def predict():
    start_time = time.time()
    loaded_model, vectorizer = load_model()
    # Check if models are loaded
    if loaded_model is None or vectorizer is None:
        if not load_model():  # Try to load models if they're not loaded
            return jsonify({'error': 'Models not initialized properly'}), 500
    
    data = request.get_json()
    text = data.get('text', '')
    is_valid, validation_message = validate_input(text)
    if not is_valid:
        return jsonify({'error': validation_message}), 400
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Transform the text and make prediction
        text_vectorized = vectorizer.transform([text])
        prediction = loaded_model.predict(text_vectorized)[0]
        if prediction != 'REAL' and prediction != 'FAKE':
            return jsonify({'error': 'response is not REAL or FAKE'}), 400
            
        response = {
            'text': text,
            'prediction': prediction,
            'processing_time': time.time() - start_time
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    application.run(port=5000, debug=True)