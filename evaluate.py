"""
Evaluation utilities — RMSE, MAE, MAPE and the comparison table logic.
"""

import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE — handles zero actuals by skipping them."""
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray, model_name: str = "") -> dict:
    """Return a dict with all three metrics for a given forecast."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    results = {
        "model": model_name,
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "MAPE": mape(actual, predicted),
    }
    logger.info(
        f"{model_name:20s} | RMSE={results['RMSE']:,.0f} | MAE={results['MAE']:,.0f} | MAPE={results['MAPE']:.2f}%"
    )
    return results


def build_comparison_table(results_list: list) -> pd.DataFrame:
    """
    Takes a list of result dicts (from evaluate_forecast) and returns a 
    nicely sorted DataFrame. Best model = lowest MAPE.
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values("MAPE").reset_index(drop=True)
    df.index = df.index + 1  # rank starts at 1
    df.index.name = "Rank"
    return df


def select_best_model(comparison_df: pd.DataFrame) -> str:
    """Return the model name with lowest MAPE."""
    best = comparison_df.iloc[0]["model"]
    logger.info(f"Best model selected: {best}")
    return best
