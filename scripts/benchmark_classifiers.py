"""
benchmark_classifiers.py

Runs a benchmarking suite against multiple ML algorithms to determine the 
optimal classifier for semantic routing based on F1 Score and Inference Latency.

Usage:
    python scripts/benchmark_classifiers.py
"""

import time
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(message)s")

EMBEDDINGS_FILE = "data/X_embeddings.npy"
LABELS_FILE = "data/y_labels.npy"

def benchmark_model(name, model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Measure inference latency on the test set
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time = time.perf_counter()
    
    # Calculate metrics
    latency_ms = ((end_time - start_time) / len(X_test)) * 1000
    # FIX: pos_label is now 1 to match the binarized labels
    score = f1_score(y_test, y_pred, pos_label=1)
    
    return name, score, latency_ms

def main():
    print("\n--- Starting Classifier Benchmark ---\n")
    X = np.load(EMBEDDINGS_FILE)
    
    # Convert string labels to binary for XGBoost compatibility
    # 1 = narrative_shift (complex), 0 = standard_action (simple)
    y_raw = np.load(LABELS_FILE)
    y = np.array([1 if label == "narrative_shift" else 0 for label in y_raw])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("Linear SVM", LinearSVC(max_iter=2000, class_weight="balanced", dual=False)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]
    
    results = []
    for name, model in models:
        name, score, latency = benchmark_model(name, model, X_train, y_train, X_test, y_test)
        results.append((name, score, latency))
        
    # Print formatted results
    print(f"{'Model':<20} | {'F1 Score':<10} | {'Latency per prediction (ms)':<15}")
    print("-" * 60)
    for name, score, latency in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:<20} | {score:<10.4f} | {latency:<15.4f}")
    print("\n-------------------------------------\n")

if __name__ == "__main__":
    main()