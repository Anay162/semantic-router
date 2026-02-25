"""
train_classifier.py

Trains a lightweight Support Vector Machine (Linear SVM) on the embedded player commands.
This script loads the vectorized data, splits it for validation, and trains 
the model. The trained weights are then serialized for use in the live routing API.

Usage:
    python scripts/train_classifier.py
"""

import os
import logging
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Configure logging for terminal visibility
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
EMBEDDINGS_FILE = "data/X_embeddings.npy"
LABELS_FILE = "data/y_labels.npy"
MODEL_OUTPUT_DIR = "models"
MODEL_OUTPUT_FILE = f"{MODEL_OUTPUT_DIR}/intent_classifier.pkl"

def main():
    logging.info("Loading embeddings and labels...")
    try:
        X = np.load(EMBEDDINGS_FILE)
        y = np.load(LABELS_FILE)
    except FileNotFoundError:
        logging.error("Data files not found. Please run embed_data.py first.")
        return
        
    logging.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} dimensions.")
    
    # Split the dataset (80% training, 20% testing)
    # Stratify ensures the class balance is maintained in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Training set: {X_train.shape[0]} samples. Testing set: {X_test.shape[0]} samples.")
    
    # Initialize and train the Linear SVM classifier
    # This was chosen empirically via our benchmark script for having the lowest latency 
    # and highest F1 score on high-dimensional dense embeddings.
    logging.info("Training Linear SVM model...")
    clf = LinearSVC(max_iter=2000, class_weight="balanced", dual="auto", random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    logging.info("Evaluating model on test set...")
    y_pred = clf.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print("-----------------------------\n")
    
    # Ensure model directory exists and save the weights
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_OUTPUT_FILE)
    logging.info(f"Model successfully saved to {MODEL_OUTPUT_FILE}")
    logging.info("Pipeline Step 3 Complete. The routing engine is ready for deployment.")

if __name__ == "__main__":
    main()