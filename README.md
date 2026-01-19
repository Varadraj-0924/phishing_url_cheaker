# Phishing URL Detector

A machine learning-based web application that detects phishing URLs using a Random Forest classifier. The application analyzes 25+ features of a URL to determine if it's legitimate or a phishing attempt.

## Features

- 🛡️ Real-time URL analysis
- 🤖 Machine learning-powered detection
- 📊 Confidence scores and probability breakdown
- 🎨 Modern, responsive web interface
- 🔍 25+ URL features analyzed

## Project Structure

```
phishing-url-detector/
│
├── model/
│   └── phishing_model.pkl          # Trained model (generated after training)
│
├── app.py                          # Flask web application
├── feature_extraction.py           # URL feature extraction logic
├── train_model.py                  # Model training script
├── phishing.csv                    # Training dataset
│
├── templates/
│   └── index.html                  # Web interface template
│
├── static/
│   └── style.css                   # Styling for web interface
│
└── requirements.txt                # Python dependencies
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

First, train the machine learning model using the provided dataset:

```bash
python train_model.py
```

This will:
- Load the dataset from `phishing.csv`
- Extract features from URLs
- Train a Random Forest classifier
- Save the model to `model/phishing_model.pkl`
- Display training accuracy and evaluation metrics

### Step 2: Run the Web Application

Start the Flask web server:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`

### Step 3: Use the Web Interface

1. Open your browser and navigate to `http://127.0.0.1:5000`
2. Enter a URL in the input field
3. Click "Check URL" to analyze it
4. View the results with confidence scores

## How It Works

The application uses machine learning to detect phishing URLs by analyzing various features:

- URL length and structure
- Domain characteristics (length, subdomains, TLD)
- Presence of suspicious keywords
- Special characters and patterns
- IP addresses in domain
- URL shortening services
- And more...

## Dataset

The `phishing.csv` file contains labeled URLs:
- `label = 0`: Legitimate URLs
- `label = 1`: Phishing URLs

You can expand the dataset by adding more URLs to improve model accuracy.

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 25 features extracted from URLs
- **Training**: 80% train, 20% test split

## Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- tldextract

## License

This project is open source and available for educational purposes.

## Disclaimer

This tool is for educational purposes only. Always use multiple security measures and verify URLs through official channels.
