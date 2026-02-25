"""
Generates a synthetic dataset of player commands for Pax Historia.
This script uses an LLM to simulate grand strategy player inputs, categorizing 
them into 'standard_action' (low compute requirements) and 'narrative_shift' 
(high compute requirements). The output is saved as a structured JSON file 
for downstream embedding and classifier training.

Usage:
    python scripts/generate_data.py
"""

import os
import json
import logging
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging for production-level visibility in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables (ensure .env is in your .gitignore)
load_dotenv()

# Initialize OpenAI Client
# Note: To use an open-source model (e.g., Llama 3 via vLLM), you can point 
# the base_url to your local or HPC endpoint.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
MODEL = "gpt-5.2"
OUTPUT_FILE = "data/raw_prompts.json"
NUM_BATCHES = 10  # Number of API calls to make
EXAMPLES_PER_BATCH = 50  # Examples generated per call

def generate_prompt_batch(batch_size: int) -> List[Dict[str, str]]:
    """
    Calls the LLM to generate a batch of structured player actions.
    Uses JSON mode to guarantee parseable output.
    """
    system_prompt = f"""
    You are an expert data generator for an AI-powered grand strategy game.
    Generate {batch_size} unique, diverse player text commands. 
    
    Exactly half must be 'standard_action': routine gameplay like moving troops, changing taxes, or basic diplomacy.
    Exactly half must be 'narrative_shift': complex, world-altering events, sci-fi twists, or custom religions.
    
    Output strictly as a JSON object with a single key "data" containing a list of objects.
    Each object must have "text" (the player's input) and "label" (either 'standard_action' or 'narrative_shift').
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the next batch of player commands."}
            ],
            temperature=0.8, # Higher temperature for creative variance
        )
        
        # Parse the JSON string returned by the model
        content = json.loads(response.choices[0].message.content)
        return content.get("data", [])
        
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return []

def main():
    """Main execution function to orchestrate data generation."""
    logging.info(f"Starting synthetic data generation using {MODEL}...")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    all_data = []
    
    for i in range(NUM_BATCHES):
        logging.info(f"Processing batch {i+1}/{NUM_BATCHES}...")
        batch_data = generate_prompt_batch(EXAMPLES_PER_BATCH)
        all_data.extend(batch_data)
        
    if all_data:
        # Save to disk
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_data, f, indent=4)
        logging.info(f"Successfully saved {len(all_data)} examples to {OUTPUT_FILE}.")
    else:
        logging.warning("No data generated. Check your API key and network connection.")

if __name__ == "__main__":
    main()