from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
import sklearn  # Required for pickle to load the model
from feature_extraction import extract_features

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/phishing_model.pkl'
model = None

def load_model():
    """Load the trained phishing detection model."""
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return True
    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Please run train_model.py first to train the model.")
        return False


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a URL is phishing or legitimate."""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'error': 'URL is required',
                'prediction': None,
                'confidence': None
            }), 400
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Extract features
        features = extract_features(url)
        feature_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.',
                'prediction': None,
                'confidence': None
            }), 500
        
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]
        
        # Get confidence score
        confidence = float(probabilities[prediction]) * 100
        
        # Map prediction to label
        label = 'Phishing' if prediction == 1 else 'Legitimate'
        
        return jsonify({
            'url': url,
            'prediction': label,
            'confidence': round(confidence, 2),
            'phishing_probability': round(probabilities[1] * 100, 2),
            'legitimate_probability': round(probabilities[0] * 100, 2)
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error processing request: {str(e)}',
            'prediction': None,
            'confidence': None
        }), 500


if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("\n" + "=" * 60)
        print("Phishing URL Detector Web App")
        print("=" * 60)
        print("Server starting on http://127.0.0.1:5000")
        print("Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nPlease run 'python train_model.py' first to train the model.")
u9