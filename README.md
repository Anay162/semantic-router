# Semantic LLM Router: Inference Cost & Latency Optimizer

## The Core Problem
As AI-native applications scale, routing every user prompt through expensive, high-latency frontier models (e.g., GPT-4o, Claude 3.5 Sonnet) creates an unsustainable inference burn rate. However, a significant portion of user interactions are routine and can be handled by much smaller, faster models (e.g., Llama-3-8B) with zero drop in perceived quality.

## The Solution
This repository contains a high-speed, local Semantic Routing Engine. It sits in front of the LLM execution layer, intercepts user commands, embeds them in real-time, and uses a trained Linear SVM to classify the semantic complexity of the intent. 

By dynamically routing "Simple" interactions to cheap/local models and reserving "Complex" interactions strictly for premium models, this architecture dramatically reduces API costs while preserving high-fidelity outputs.

### Technical Architecture
1. **Data Engine**: Synthetic dataset generation via OpenAI API.
2. **Embedding Layer**: `all-MiniLM-L6-v2` via `sentence-transformers` (Runs locally, executes in milliseconds).
3. **Classification Engine**: A custom-trained Linear SVM. (Benchmarked against XGBoost and Random Forest; chosen specifically for sub-50ms inference latency and perfect linear separability on dense embeddings).
4. **API Gateway**: Asynchronous FastAPI backend.

## Quickstart: Docker (Recommended)
To test the routing logic locally without configuring a Python environment, run the pre-configured Docker container:

```bash
docker build -t semantic-router .
docker run -p 8000:8000 semantic-router
```

Once the container is running, open a new terminal window and test the classification logic via a curl request:
curl -X 'POST' \
  'http://127.0.0.1:8000/route' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Calculate the basic tax rate for this region."}'

## Local Setup & Development
If you prefer to run the training pipelines, view the benchmark scripts, or open the interactive Jupyter Notebook locally without Docker:

# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (macOS/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API locally
python -m uvicorn app.main:app --reload
