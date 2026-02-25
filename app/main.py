"""
app/main.py
The FastAPI application entry point.
"""
from fastapi import FastAPI
from app.schemas import RouteRequest, RouteResponse
from app.engine import router_instance

app = FastAPI(
    title="Semantic Router",
    description="An intelligent routing layer to drastically reduce LLM inference costs.",
    version="1.0.0"
)

@app.post("/route", response_model=RouteResponse)
async def route_player_action(request: RouteRequest):
    """
    Intercepts a player command, vectorizes it, classifies the complexity, 
    and routes it to the most cost-effective LLM in milliseconds.
    """
    # Call the engine's routing logic
    result = router_instance.route_query(request.text)
    
    # Return the formatted response matching our Pydantic schema
    return RouteResponse(**result)

@app.get("/health")
async def health_check():
    return {"status": "operational", "models_loaded": True}