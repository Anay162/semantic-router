"""
Transforms raw JSON text prompts into dense vector embeddings.
This script utilizes a lightweight, local sentence-transformer model 
(all-MiniLM-L6-v2) to vectorize player commands. The resulting embeddings 
and their corresponding labels are saved as NumPy arrays for fast, 
memory-efficient loading during classifier training.

Usage:
    python scripts/embed_data.py
"""

import os
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
INPUT_FILE = "data/raw_prompts.json"
EMBEDDINGS_OUTPUT = "data/X_embeddings.npy"
LABELS_OUTPUT = "data/y_labels.npy"
MODEL_NAME = "all-MiniLM-L6-v2" # Lightweight, runs efficiently on CPU

def load_raw_data(filepath: str) -> tuple[list[str], list[str]]:
    """Loads text and labels from the generated JSON dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at {filepath}. Run 01_generate_data.py first.")
        
    with open(filepath, "r") as f:
        data = json.load(f)
        
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels

def main():
    logging.info(f"Loading raw data from {INPUT_FILE}...")
    try:
        texts, labels = load_raw_data(INPUT_FILE)
    except FileNotFoundError as e:
        logging.error(e)
        return

    logging.info(f"Loaded {len(texts)} samples. Initializing embedding model '{MODEL_NAME}'...")
    
    # Load the embedding model (downloads automatically on first run)
    model = SentenceTransformer(MODEL_NAME)
    
    # Generate embeddings
    # show_progress_bar provides good terminal feedback for larger datasets
    logging.info("Vectorizing text. This may take a minute depending on your CPU...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(EMBEDDINGS_OUTPUT), exist_ok=True)
    
    # Save the arrays to disk
    logging.info("Saving embeddings and labels to disk...")
    np.save(EMBEDDINGS_OUTPUT, embeddings)
    np.save(LABELS_OUTPUT, np.array(labels))
    
    logging.info(f"Successfully saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")
    logging.info("Pipeline Step 2 Complete. Ready for model training.")

if __name__ == "__main__":
    main()