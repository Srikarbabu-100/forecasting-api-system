"""
main.py — project entry point.

Usage:
    # Train all states (takes a while)
    python main.py --train

    # Train specific states (faster for testing)
    python main.py --train --state California Texas

    # Generate a forecast (models must be trained first)
    python main.py --forecast --state Texas

    # Start the API server
    python main.py --serve

    # Run EDA + visualizations
    python main.py --explore
"""

import argparse
import sys
import json


def run_training(states):
    from src.train import run_training
    run_training(states_to_train=states if states else None)


def run_forecast(state):
    from src.predict import generate_forecast
    result = generate_forecast(state=state)
    print(json.dumps(result, indent=2))


def run_server():
    import uvicorn
    print("Starting API server at http://localhost:8000")
    print("Docs at http://localhost:8000/docs")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)


def run_exploration():
    from src.data_preprocessing import run_preprocessing_pipeline
    from src.visualize import plot_weekly_sales_overview

    print("Running data exploration...")
    state_data = run_preprocessing_pipeline()

    # summary stats
    print(f"\nDataset Summary:")
    print(f"  Total states: {len(state_data)}")
    for state, series in list(state_data.items())[:5]:
        print(f"  {state}: {len(series)} weeks | {series.index.min().date()} → {series.index.max().date()} | mean={series.mean():,.0f}")
    print("  ...")

    # save overview plot
    plot_weekly_sales_overview(state_data)
    print("\nOverview plot saved to outputs/states_overview.png")


def run_visualize_after_training(state):
    """Generate comparison plots for a state that was already trained."""
    from src.data_preprocessing import run_preprocessing_pipeline, get_train_test_split
    from src.evaluate import evaluate_forecast, build_comparison_table
    from src.visualize import plot_forecast_vs_actual, plot_model_comparison, plot_feature_importance
    from src.config import TEST_WEEKS

    from src.models.arima_model import load_sarima, predict_sarima
    from src.models.prophet_model import load_prophet, predict_prophet
    from src.models.xgboost_model import load_xgboost, predict_xgboost, get_feature_importance
    from src.models.lstm_model import load_lstm, predict_lstm

    import numpy as np

    state_data = run_preprocessing_pipeline()
    series = state_data[state]
    train, test = get_train_test_split(series, TEST_WEEKS)

    preds = {}
    results = []

    for name, load_fn, pred_fn in [
        ("SARIMA", load_sarima, predict_sarima),
        ("Prophet", load_prophet, predict_prophet),
        ("XGBoost", load_xgboost, predict_xgboost),
        ("LSTM", load_lstm, predict_lstm),
    ]:
        try:
            m = load_fn(state)
            p = pred_fn(m, steps=len(test))
            preds[name] = p
            results.append(evaluate_forecast(test.values, p, name))
        except Exception as e:
            print(f"Could not load {name}: {e}")

    if preds:
        comparison = build_comparison_table(results)
        plot_forecast_vs_actual(train, test, preds, state)
        plot_model_comparison(comparison, state)

    # feature importance for XGBoost
    try:
        xgb = load_xgboost(state)
        fi = get_feature_importance(xgb)
        plot_feature_importance(fi, state)
    except Exception as e:
        print(f"XGBoost feature importance failed: {e}")

    print(f"Plots saved to outputs/{state}_*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Forecasting System")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--forecast", action="store_true", help="Generate forecast")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--explore", action="store_true", help="Run data exploration")
    parser.add_argument("--visualize", action="store_true", help="Generate plots for a state")
    parser.add_argument(
        "--state", nargs="*", default=None,
        help="State(s) to process. Used with --train, --forecast, --visualize"
    )

    args = parser.parse_args()

    if not any([args.train, args.forecast, args.serve, args.explore, args.visualize]):
        parser.print_help()
        sys.exit(0)

    if args.train:
        run_training(args.state)

    if args.forecast:
        if not args.state:
            print("Error: --forecast requires --state <StateName>")
            sys.exit(1)
        for s in args.state:
            run_forecast(s)

    if args.visualize:
        if not args.state:
            print("Error: --visualize requires --state <StateName>")
            sys.exit(1)
        for s in args.state:
            run_visualize_after_training(s)

    if args.explore:
        run_exploration()

    if args.serve:
        run_server()
