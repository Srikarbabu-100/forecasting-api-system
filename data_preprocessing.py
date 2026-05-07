"""
Data preprocessing pipeline.

Steps:
  1. Load raw Excel
  2. Parse and clean dates
  3. Aggregate to weekly level per state
  4. Fill missing weeks (forward fill + interpolation)
  5. Return a clean dict of {state -> pd.Series}
"""

import pandas as pd
import numpy as np
from src.config import RAW_DATA_FILE, FREQ
from src.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(filepath: str = RAW_DATA_FILE) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_excel(filepath)
    logger.info(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    return df


def clean_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning — fix dtypes, drop obvious junk."""
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # drop rows where we genuinely don't know the date
    bad_dates = df["Date"].isna().sum()
    if bad_dates > 0:
        logger.warning(f"Dropping {bad_dates} rows with unparseable dates")
        df = df.dropna(subset=["Date"])

    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df["State"] = df["State"].astype(str).str.strip()

    # sanity check — negative sales don't really make sense here
    neg_rows = (df["Total"] < 0).sum()
    if neg_rows > 0:
        logger.warning(f"Found {neg_rows} rows with negative Total — clipping to 0")
        df["Total"] = df["Total"].clip(lower=0)

    return df


def aggregate_weekly(df: pd.DataFrame) -> dict:
    """
    Group by State, resample to weekly frequency, sum sales.
    Returns a dict {state: pd.Series} with DatetimeIndex.
    """
    state_series = {}
    states = df["State"].unique()
    logger.info(f"Aggregating {len(states)} states to weekly frequency")

    for state in states:
        state_df = df[df["State"] == state].copy()
        state_df = state_df.set_index("Date").sort_index()

        # sum up multiple transactions on same/nearby dates
        weekly = state_df["Total"].resample(FREQ).sum()

        # fill weeks that had zero transactions with NaN first, then interpolate
        # using a full date range avoids gaps confusing the models later
        full_range = pd.date_range(
            start=weekly.index.min(), end=weekly.index.max(), freq=FREQ
        )
        weekly = weekly.reindex(full_range)

        # interpolate short gaps, forward-fill the rest
        weekly = weekly.interpolate(method="linear", limit=4)
        weekly = weekly.ffill().bfill()

        # still some NaNs? just drop — shouldn't happen but just in case
        weekly = weekly.dropna()

        state_series[state] = weekly

    logger.info("Weekly aggregation done")
    return state_series


def get_train_test_split(series: pd.Series, test_weeks: int):
    """Time-series safe split. No shuffling, no leakage."""
    if len(series) <= test_weeks:
        raise ValueError(
            f"Series too short ({len(series)} weeks) for test_weeks={test_weeks}"
        )
    train = series.iloc[:-test_weeks]
    test = series.iloc[-test_weeks:]
    return train, test


def run_preprocessing_pipeline():
    """Entry point to run the whole thing end to end."""
    raw = load_raw_data()
    clean = clean_and_parse(raw)
    state_data = aggregate_weekly(clean)

    # quick summary
    lengths = [len(s) for s in state_data.values()]
    logger.info(
        f"Preprocessing complete. States: {len(state_data)}, "
        f"avg weeks per state: {np.mean(lengths):.1f}, "
        f"min: {min(lengths)}, max: {max(lengths)}"
    )
    return state_data


if __name__ == "__main__":
    data = run_preprocessing_pipeline()
    for state, series in list(data.items())[:3]:
        print(f"{state}: {len(series)} weeks, {series.index.min().date()} to {series.index.max().date()}")
