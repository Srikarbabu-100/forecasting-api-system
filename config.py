"""
Central config file — keeps all the magic numbers and paths in one place
so I don't have to hunt through 10 files when something changes.
"""

import os

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

RAW_DATA_FILE = os.path.join(DATA_DIR, "sales_data.xlsx")

# forecasting params
FORECAST_HORIZON = 8   # weeks ahead
FREQ = "W"             # weekly aggregation
TEST_WEEKS = 12        # hold-out set size for evaluation

# feature engineering
LAG_FEATURES = [1, 7, 30]   # lag in days (after resampling to weekly, we'll adapt)
ROLLING_WINDOWS = [4, 8]    # rolling stats window sizes (in weeks)

# model params (kept simple on purpose)
ARIMA_ORDER = (1, 1, 1)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 0, 52)  # yearly seasonality

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

LSTM_LOOKBACK = 12       # weeks of history for LSTM input
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 16
LSTM_UNITS = 64

# evaluation
METRICS = ["RMSE", "MAE", "MAPE"]

# make sure output dirs exist when config is imported
for _dir in [MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
