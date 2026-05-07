# forecasting-api-system
# Sales Forecasting System

**8-week ahead weekly sales forecasting for US states using multiple ML models**

---

## Project Overview

This project builds a complete time series forecasting pipeline for retail sales data. Given historical weekly sales per US state, the system trains four different models, evaluates them on a held-out test set, automatically selects the best performer, and exposes predictions via a REST API.

### Why four models?
No single model wins everywhere. SARIMA handles linear trends and seasonality well. Prophet is great at picking up holiday effects. XGBoost is often the most reliable for tabular/lag-based features. LSTM can theoretically capture long-range patterns — though it's the most data-hungry.

### Dataset
- **Source:** `data/sales_data.xlsx`
- **Coverage:** 43 US states, weekly beverage sales, 2019–2023
- **Structure:** State | Date | Total | Category

---

## Project Structure

```
time-series-forecasting-system/
│
├── data/
│   └── sales_data.xlsx            # Raw data
│
├── notebooks/
│   └── exploratory_analysis.py    # EDA script
│
├── src/
│   ├── config.py                  # All constants and paths
│   ├── logger.py                  # Logging setup
│   ├── data_preprocessing.py      # Load → clean → weekly aggregation
│   ├── feature_engineering.py     # Lag features, rolling stats, holidays
│   ├── evaluate.py                # RMSE, MAE, MAPE + comparison table
│   ├── train.py                   # Training orchestrator (all models, all states)
│   ├── predict.py                 # Inference pipeline
│   ├── visualize.py               # All plotting functions
│   └── models/
│       ├── arima_model.py         # SARIMA
│       ├── prophet_model.py       # Facebook Prophet
│       ├── xgboost_model.py       # XGBoost (lag features → supervised)
│       └── lstm_model.py          # LSTM (TensorFlow/Keras)
│
├── api/
│   └── app.py                     # FastAPI REST API
│
├── models/                        # Saved models (auto-created)
├── outputs/                       # Plots + CSV reports (auto-created)
├── logs/                          # Log files (auto-created)
│
├── main.py                        # CLI entry point
├── requirements.txt
└── README.md
```

---

## Architecture

```
Raw Excel Data
      │
      ▼
Data Preprocessing
(parse dates → weekly resample → fill gaps)
      │
      ▼
Feature Engineering (for XGBoost)
(lags, rolling stats, calendar, holiday flags)
      │
      ├──► SARIMA training
      ├──► Prophet training
      ├──► XGBoost training
      └──► LSTM training
             │
             ▼
        Model Evaluation
        (RMSE / MAE / MAPE on hold-out set)
             │
             ▼
        Best Model Selection (lowest MAPE)
             │
             ▼
        FastAPI REST API
        POST /forecast → 8-week predictions
```

---

## Installation

```bash
# Clone / download the project
cd time-series-forecasting-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# or
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**Note:** TensorFlow installation can be slow. On Apple Silicon, use `tensorflow-macos` instead.

---

## Execution Steps

### Step 1 — Explore the data (optional but recommended)
```bash
python notebooks/exploratory_analysis.py
```
This generates EDA plots in `outputs/` and prints a data summary.

### Step 2 — Train models

**Train all 43 states** (takes ~30–60 min depending on hardware):
```bash
python main.py --train
```

**Train specific states** (much faster, good for testing):
```bash
python main.py --train --state California Texas "New York"
```

Training saves:
- Model files to `models/`
- Per-state comparison CSVs to `outputs/`
- Overall summary to `outputs/training_summary.csv`

### Step 3 — Generate a forecast (CLI)
```bash
python main.py --forecast --state Texas
```

### Step 4 — Generate plots for a state
```bash
python main.py --visualize --state California
```
Creates:
- `outputs/California_forecast_vs_actual.png`
- `outputs/California_model_comparison.png`
- `outputs/California_feature_importance.png`

### Step 5 — Start the API
```bash
python main.py --serve
```
API runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API Usage

### POST /forecast

**Request:**
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"state": "Texas"}'
```

**Response:**
```json
{
  "state": "Texas",
  "model_used": "XGBoost",
  "forecast_horizon_weeks": 8,
  "generated_at": "2024-01-15T10:30:00",
  "forecast": [
    {"week": 1, "date": "2024-01-21", "predicted_sales": 125000000.00},
    {"week": 2, "date": "2024-01-28", "predicted_sales": 128500000.00},
    ...
  ]
}
```

**With custom model:**
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"state": "California", "model": "Prophet", "horizon": 12}'
```

### GET /states
```bash
curl http://localhost:8000/states
```

### GET /model-info/{state}
```bash
curl http://localhost:8000/model-info/California
```

---

## Feature Engineering Details

For XGBoost, the following features are built from the weekly sales series:

| Feature | Description |
|---|---|
| `lag_1` | Sales from 1 week ago |
| `lag_4` | Sales from ~1 month ago |
| `lag_8` | Sales from ~2 months ago |
| `lag_52` | Sales from same week last year |
| `rolling_mean_4` | 4-week moving average |
| `rolling_mean_8` | 8-week moving average |
| `rolling_std_4` | 4-week sales volatility |
| `rolling_std_8` | 8-week sales volatility |
| `month` | Month (1-12) |
| `quarter` | Quarter (1-4) |
| `week_of_year` | ISO week number |
| `day_of_week` | Day of week (0=Monday) |
| `year` | Calendar year |
| `is_q4` | Is Q4 (holiday season flag) |
| `holiday_flag` | US federal holiday in this week |

**Data leakage prevention:** All lag/rolling features use `shift(1)` before rolling computation. The train/test split is strictly temporal.

---

## Model Evaluation

Metrics used:

| Metric | Formula | Notes |
|---|---|---|
| RMSE | √(mean((actual - pred)²)) | Penalizes large errors |
| MAE | mean(|actual - pred|) | Robust to outliers |
| MAPE | mean(|actual - pred| / actual) × 100 | Interpretable % error |

Best model = lowest MAPE on the hold-out test set (last 12 weeks).

---

## Screenshots

*After training, find plots in the `outputs/` folder:*

- `outputs/states_overview.png` — overview of all states
- `outputs/eda_*.png` — EDA visualizations
- `outputs/<state>_forecast_vs_actual.png` — model predictions vs actual
- `outputs/<state>_model_comparison.png` — metric comparison bar charts
- `outputs/<state>_feature_importance.png` — XGBoost feature importance

---

## Configuration

All key parameters are in `src/config.py`:

```python
FORECAST_HORIZON = 8     # weeks ahead
TEST_WEEKS = 12          # hold-out set size
LSTM_LOOKBACK = 12       # weeks of history for LSTM
LSTM_EPOCHS = 50
XGBOOST_PARAMS = { "n_estimators": 200, ... }
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 0, 52)
```

---

## Future Improvements

Things I'd add given more time:

1. **Hyperparameter tuning** — ARIMA auto-order selection (auto_arima), XGBoost grid search
2. **Ensemble forecasting** — weighted blend of model predictions
3. **Exogenous variables** — add weather, economic indicators, promo flags
4. **Online learning** — update models as new data arrives without full retraining
5. **Uncertainty quantification** — prediction intervals for all models (not just Prophet)
6. **Database backend** — store forecasts in Postgres instead of flat files
7. **Dockerization** — containerize for easy deployment
8. **Model monitoring** — track prediction drift over time

---

## Dependencies

Key packages:
- **statsmodels** — SARIMA
- **prophet** — Facebook Prophet
- **xgboost** — XGBoost
- **tensorflow** — LSTM
- **fastapi + uvicorn** — REST API
- **pandas, numpy, scikit-learn** — data processing
- **matplotlib, seaborn** — visualizations

See `requirements.txt` for exact versions.
