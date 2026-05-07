"""
Training orchestrator.

Loops through all states, trains all 4 models on each, evaluates on
the held-out test set, selects the best model, and saves everything.

This can take a while — mainly LSTM training + SARIMA fitting.
Run with a single state argument for quick testing:
    python -m src.train --state California
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from src.data_preprocessing import run_preprocessing_pipeline, get_train_test_split
from src.evaluate import evaluate_forecast, build_comparison_table, select_best_model
from src.config import FORECAST_HORIZON, TEST_WEEKS, OUTPUTS_DIR, MODELS_DIR
from src.logger import get_logger

# model imports
from src.models.arima_model import train_sarima, predict_sarima, save_sarima
from src.models.prophet_model import train_prophet, predict_prophet, save_prophet
from src.models.xgboost_model import train_xgboost, predict_xgboost, save_xgboost
from src.models.lstm_model import train_lstm, predict_lstm, save_lstm

logger = get_logger(__name__)


def train_state(state: str, series: pd.Series) -> dict:
    """Train all models for one state and evaluate on test set."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training models for: {state} ({len(series)} weeks of data)")
    logger.info(f"{'='*60}")

    # time-series safe split
    try:
        train, test = get_train_test_split(series, TEST_WEEKS)
    except ValueError as e:
        logger.warning(f"[{state}] Skipping: {e}")
        return None

    results = []

    # ---- SARIMA ----
    try:
        sarima_result = train_sarima(train, state)
        sarima_preds = predict_sarima(sarima_result, steps=len(test))
        results.append(evaluate_forecast(test.values, sarima_preds, "SARIMA"))
        save_sarima(sarima_result, state)
    except Exception as e:
        logger.error(f"[{state}] SARIMA failed: {e}")
        results.append({"model": "SARIMA", "RMSE": np.inf, "MAE": np.inf, "MAPE": np.inf})

    # ---- Prophet ----
    try:
        prophet_result = train_prophet(train, state)
        prophet_preds = predict_prophet(prophet_result, steps=len(test))
        results.append(evaluate_forecast(test.values, prophet_preds, "Prophet"))
        save_prophet(prophet_result, state)
    except Exception as e:
        logger.error(f"[{state}] Prophet failed: {e}")
        results.append({"model": "Prophet", "RMSE": np.inf, "MAE": np.inf, "MAPE": np.inf})

    # ---- XGBoost ----
    try:
        xgb_result = train_xgboost(train, state)
        xgb_preds = predict_xgboost(xgb_result, steps=len(test))
        results.append(evaluate_forecast(test.values, xgb_preds, "XGBoost"))
        save_xgboost(xgb_result, state)
    except Exception as e:
        logger.error(f"[{state}] XGBoost failed: {e}")
        results.append({"model": "XGBoost", "RMSE": np.inf, "MAE": np.inf, "MAPE": np.inf})

    # ---- LSTM ----
    try:
        lstm_result = train_lstm(train, state)
        lstm_preds = predict_lstm(lstm_result, steps=len(test))
        results.append(evaluate_forecast(test.values, lstm_preds, "LSTM"))
        save_lstm(lstm_result, state)
    except Exception as e:
        logger.error(f"[{state}] LSTM failed: {e}")
        results.append({"model": "LSTM", "RMSE": np.inf, "MAE": np.inf, "MAPE": np.inf})

    # comparison table + best model selection
    comparison = build_comparison_table(results)
    best_model = select_best_model(comparison)

    print(f"\n[{state}] Model Comparison:")
    print(comparison.to_string())
    print(f"  => Best model: {best_model}\n")

    # save comparison table for this state
    comp_path = os.path.join(OUTPUTS_DIR, f"{state}_model_comparison.csv")
    comparison.to_csv(comp_path)

    # save best model name so API can look it up quickly
    best_path = os.path.join(MODELS_DIR, f"{state}_best_model.json")
    with open(best_path, "w") as f:
        json.dump({"state": state, "best_model": best_model}, f)

    return {
        "state": state,
        "comparison": comparison,
        "best_model": best_model,
        "train_size": len(train),
        "test_size": len(test),
    }


def run_training(states_to_train: list = None):
    """Main training loop."""
    all_state_data = run_preprocessing_pipeline()

    if states_to_train:
        # filter to requested states
        not_found = [s for s in states_to_train if s not in all_state_data]
        if not_found:
            logger.warning(f"States not found in data: {not_found}")
        all_state_data = {s: v for s, v in all_state_data.items() if s in states_to_train}

    logger.info(f"Training {len(all_state_data)} states")

    all_results = []
    failed = []
    for state, series in all_state_data.items():
        result = train_state(state, series)
        if result:
            all_results.append(result)
        else:
            failed.append(state)

    if failed:
        logger.warning(f"Failed states: {failed}")

    # aggregate summary
    if all_results:
        summary_rows = []
        for r in all_results:
            row = {"state": r["state"], "best_model": r["best_model"]}
            # grab the best model's metrics
            best_row = r["comparison"][r["comparison"]["model"] == r["best_model"]].iloc[0]
            row.update(best_row.to_dict())
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(OUTPUTS_DIR, "training_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Training summary saved to {summary_path}")
        print("\n=== TRAINING SUMMARY ===")
        print(summary_df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument(
        "--state",
        nargs="*",
        default=None,
        help="State(s) to train on. If not provided, trains all states.",
    )
    args = parser.parse_args()

    run_training(states_to_train=args.state)
