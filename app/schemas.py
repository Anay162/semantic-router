"""
app/schemas.py
Pydantic models for API request and response validation.
"""
from pydantic import BaseModel, Field

class RouteRequest(BaseModel):
    text: str = Field(..., description="The raw text command from the Pax Historia player.")

class RouteResponse(BaseModel):
    action_type: str = Field(..., description="The classified intent of the user's action.")
    target_model: str = Field(..., description="The LLM recommended for this specific query.")
    estimated_cost_per_1m: str = Field(..., description="The estimated cost per 1M tokens for the target model.")
    routing_latency_ms: float = Field(..., description="How long the semantic router took to execute.")