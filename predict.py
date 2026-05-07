"""
Prediction pipeline.

Loads saved models for a given state and generates 8-week forecasts.
Handles the case where models haven't been trained yet — returns a 
clear error rather than crashing the API.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import MODELS_DIR, FORECAST_HORIZON
from src.data_preprocessing import run_preprocessing_pipeline
from src.logger import get_logger

from src.models.arima_model import load_sarima, predict_sarima
from src.models.prophet_model import load_prophet, predict_prophet
from src.models.xgboost_model import load_xgboost, predict_xgboost
from src.models.lstm_model import load_lstm, predict_lstm

logger = get_logger(__name__)


def get_best_model_name(state: str) -> str:
    """Read the best model selection saved during training."""
    path = os.path.join(MODELS_DIR, f"{state}_best_model.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained models found for '{state}'. Please run training first."
        )
    with open(path) as f:
        data = json.load(f)
    return data["best_model"]


def _load_model(state: str, model_name: str) -> tuple:
    """Load the right model based on name. Returns (model_result, predict_fn)."""
    loaders = {
        "SARIMA": (load_sarima, predict_sarima),
        "Prophet": (load_prophet, predict_prophet),
        "XGBoost": (load_xgboost, predict_xgboost),
        "LSTM": (load_lstm, predict_lstm),
    }

    if model_name not in loaders:
        raise ValueError(f"Unknown model type: {model_name}")

    load_fn, predict_fn = loaders[model_name]
    model_result = load_fn(state)
    return model_result, predict_fn


def generate_forecast(state: str, model_name: str = None, horizon: int = FORECAST_HORIZON) -> dict:
    """
    Main prediction function. Used by the API.
    
    Args:
        state: e.g. "Texas"
        model_name: which model to use (None = auto-select best)
        horizon: number of weeks to forecast
    
    Returns:
        dict with state, model_used, forecast_dates, and forecast values
    """
    if model_name is None:
        model_name = get_best_model_name(state)
        logger.info(f"[{state}] Auto-selected best model: {model_name}")

    logger.info(f"[{state}] Generating {horizon}-week forecast using {model_name}")

    model_result, predict_fn = _load_model(state, model_name)
    preds = predict_fn(model_result, steps=horizon)

    # figure out the forecast date range
    # we need the last date in the training data to anchor the future dates
    state_data = run_preprocessing_pipeline()
    if state not in state_data:
        raise ValueError(f"State '{state}' not found in dataset")

    last_train_date = state_data[state].index[-1]
    forecast_dates = pd.date_range(
        start=last_train_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W",
    )

    forecast_list = [
        {
            "week": i + 1,
            "date": date.strftime("%Y-%m-%d"),
            "predicted_sales": round(float(pred), 2),
        }
        for i, (date, pred) in enumerate(zip(forecast_dates, preds))
    ]

    result = {
        "state": state,
        "model_used": model_name,
        "forecast_horizon_weeks": horizon,
        "generated_at": datetime.utcnow().isoformat(),
        "forecast": forecast_list,
    }

    logger.info(f"[{state}] Forecast complete. Range: {preds.min():,.0f} - {preds.max():,.0f}")
    return result


def get_all_trained_states() -> list:
    """Return list of states that have trained models."""
    states = []
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith("_best_model.json"):
            state = fname.replace("_best_model.json", "")
            states.append(state)
    return sorted(states)
