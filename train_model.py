import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from feature_extraction import extract_features, get_feature_names


def load_data(csv_file='phishing.csv'):
    """
    Load dataset from CSV file.
    Expected format: url,label (where label is 0 for legitimate, 1 for phishing)
    """
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Creating sample dataset...")
        create_sample_dataset(csv_file)
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} URLs from {csv_file}")
    return df


def create_sample_dataset(csv_file):
    """
    Create a sample dataset if one doesn't exist.
    """
    sample_urls = [
        # Legitimate URLs (label = 0)
        ('https://www.google.com', 0),
        ('https://www.github.com', 0),
        ('https://www.microsoft.com', 0),
        ('https://www.amazon.com', 0),
        ('https://www.facebook.com', 0),
        ('https://www.twitter.com', 0),
        ('https://www.linkedin.com', 0),
        ('https://www.youtube.com', 0),
        ('https://www.wikipedia.org', 0),
        ('https://www.stackoverflow.com', 0),
        ('https://www.reddit.com', 0),
        ('https://www.netflix.com', 0),
        ('https://www.apple.com', 0),
        ('https://www.python.org', 0),
        ('https://www.docker.com', 0),
        
        # Phishing URLs (label = 1)
        ('http://secure-account-update.verify-bank-login.com', 1),
        ('https://www-paypal-account-verify.secure-update.com/login', 1),
        ('http://update-your-account-now.click-here.verify.com', 1),
        ('https://ebay-account-confirm.secure-signin.com', 1),
        ('http://bank-account-verify.update-now.com/login', 1),
        ('https://www-facebook-account-update.verify-now.com', 1),
        ('http://secure-login-verify.account-update.com', 1),
        ('https://update-account-verify.click-here.com', 1),
        ('http://bank-verify-account.secure-update.com', 1),
        ('https://account-verify-update.secure-login.com', 1),
        ('http://verify-account-update.secure-bank.com', 1),
        ('https://update-verify-account.secure-login.com', 1),
        ('http://account-update-verify.secure-bank.com', 1),
        ('https://verify-update-account.secure-login.com', 1),
        ('http://secure-verify-update.account-bank.com', 1),
    ]
    
    df = pd.DataFrame(sample_urls, columns=['url', 'label'])
    df.to_csv(csv_file, index=False)
    print(f"Created sample dataset with {len(df)} URLs")


def prepare_features(df):
    """
    Extract features from URLs.
    """
    print("Extracting features from URLs...")
    features = []
    labels = []
    
    for idx, row in df.iterrows():
        try:
            url = row['url']
            label = row['label']
            
            feature_vector = extract_features(url)
            features.append(feature_vector)
            labels.append(label)
        except Exception as e:
            print(f"Error processing URL {idx}: {e}")
            continue
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Extracted features for {len(features)} URLs")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y


def train_model(X, y):
    """
    Train Random Forest classifier.
    """
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model


def save_model(model, filepath='model/phishing_model.pkl'):
    """
    Save trained model to file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {filepath}")


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("Phishing URL Detector - Model Training")
    print("=" * 60)
    
    # Load data
    df = load_data('phishing.csv')
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
