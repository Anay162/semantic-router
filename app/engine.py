"""
app/engine.py
Loads the ML models and executes the routing logic.
"""
import time
import joblib
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("uvicorn.error")

class SemanticRouter:
    def __init__(self):
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.classifier_path = "models/intent_classifier.pkl"
        
        logger.info(f"Loading Embedding Model: {self.embedding_model_name}...")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        
        logger.info(f"Loading Classifier Model: {self.classifier_path}...")
        self.classifier = joblib.load(self.classifier_path)
        logger.info("Semantic Router successfully initialized.")

    def route_query(self, text: str) -> dict:
        start_time = time.perf_counter()
        
        # 1. Embed the text
        vector = self.embedder.encode([text])
        
        # 2. Predict the intent
        prediction = self.classifier.predict(vector)[0]
        
        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)
        
        # 3. Determine the routing logic based on the SVM's output
        if prediction == "standard_action":
            target = "Llama-3-8B (Local/Fast)"
            cost = "$0.20"
        else:
            target = "Claude-3.5-Sonnet (Premium)"
            cost = "$3.00"
            
        return {
            "action_type": prediction,
            "target_model": target,
            "estimated_cost_per_1m": cost,
            "routing_latency_ms": latency_ms
        }

# Instantiate a single global instance to be used by the API
router_instance = SemanticRouter()