"""
FastAPI application for serving forecasts.

Endpoints:
    POST /forecast          — main forecast endpoint
    GET  /states            — list states with trained models
    GET  /health            — basic health check
    GET  /model-info/{state} — what model was selected for this state

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import traceback

from src.predict import generate_forecast, get_all_trained_states, get_best_model_name
from src.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Sales Forecasting API",
    description="8-week ahead weekly sales forecasts per US state",
    version="1.0.0",
)

# CORS — allow all origins for now (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- request / response schemas ----

class ForecastRequest(BaseModel):
    state: str = Field(..., example="Texas", description="US state name")
    model: Optional[str] = Field(
        None,
        example=None,
        description="Model to use: SARIMA, Prophet, XGBoost, LSTM. Leave null for auto-select."
    )
    horizon: Optional[int] = Field(
        8, ge=1, le=26,
        description="Number of weeks to forecast (1-26). Default is 8."
    )


class WeeklyForecast(BaseModel):
    week: int
    date: str
    predicted_sales: float


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast_horizon_weeks: int
    generated_at: str
    forecast: List[WeeklyForecast]


# ---- endpoints ----

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Forecasting API is running"}


@app.get("/states")
def list_states():
    """Return all states that have trained models."""
    states = get_all_trained_states()
    return {"trained_states": states, "count": len(states)}


@app.get("/model-info/{state}")
def get_model_info(state: str):
    """Check which model was selected as best for a given state."""
    try:
        best = get_best_model_name(state)
        return {"state": state, "best_model": best}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """
    Generate sales forecast for the given state.
    
    Example request:
    ```json
    {
        "state": "Texas"
    }
    ```
    
    Example response:
    ```json
    {
        "state": "Texas",
        "model_used": "XGBoost",
        "forecast_horizon_weeks": 8,
        "generated_at": "2024-01-15T10:30:00",
        "forecast": [
            {"week": 1, "date": "2024-01-21", "predicted_sales": 125000000.0},
            ...
        ]
    }
    ```
    """
    logger.info(f"Forecast request: state={request.state}, model={request.model}, horizon={request.horizon}")

    try:
        result = generate_forecast(
            state=request.state,
            model_name=request.model,
            horizon=request.horizon,
        )
        return ForecastResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Models not found for state '{request.state}'. "
                   f"Run training first: python main.py --train --state '{request.state}'"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error for {request.state}: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error. Check logs.")


@app.get("/")
def root():
    return {
        "message": "Sales Forecasting API",
        "docs": "/docs",
        "endpoints": ["/health", "/states", "/forecast", "/model-info/{state}"],
    }
