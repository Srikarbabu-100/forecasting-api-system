"""
Feature engineering for tree-based and regression models.

Creates lag features, rolling stats, calendar features, and holiday flags.
IMPORTANT: all features are engineered without leaking future data — 
shift() ensures we only use past values.
"""

import pandas as pd
import numpy as np
import holidays
from src.logger import get_logger

logger = get_logger(__name__)

# US federal holidays — good enough for a national retail dataset
US_HOLIDAYS = holidays.US()


def is_holiday_week(date: pd.Timestamp) -> int:
    """Check if any day in the week around this date is a US federal holiday."""
    for offset in range(-3, 4):
        if (date + pd.Timedelta(days=offset)) in US_HOLIDAYS:
            return 1
    return 0


def build_features(series: pd.Series, label: str = "sales") -> pd.DataFrame:
    """
    Takes a weekly time series and returns a feature matrix.
    All features respect temporal ordering — no leakage.
    
    Args:
        series: pd.Series with DatetimeIndex, weekly frequency
        label: column name for the target variable
    
    Returns:
        pd.DataFrame with features + target
    """
    df = pd.DataFrame({label: series})
    df.index.name = "date"

    # --- lag features ---
    # lag_1 = last week's sales, lag_4 = a month ago, lag_52 = same week last year
    df["lag_1"] = df[label].shift(1)
    df["lag_4"] = df[label].shift(4)   # roughly a month in weekly data
    df["lag_8"] = df[label].shift(8)   # 2 months back
    df["lag_52"] = df[label].shift(52) # year-over-year reference

    # --- rolling stats ---
    # using min_periods so early rows aren't all NaN
    df["rolling_mean_4"] = df[label].shift(1).rolling(window=4, min_periods=2).mean()
    df["rolling_mean_8"] = df[label].shift(1).rolling(window=8, min_periods=4).mean()
    df["rolling_std_4"] = df[label].shift(1).rolling(window=4, min_periods=2).std()
    df["rolling_std_8"] = df[label].shift(1).rolling(window=8, min_periods=4).std()

    # --- calendar features ---
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["day_of_week"] = df.index.dayofweek  # 0=Monday
    df["year"] = df.index.year

    # is it Q4 (holiday shopping season)?
    df["is_q4"] = (df.index.quarter == 4).astype(int)

    # --- holiday flag ---
    df["holiday_flag"] = df.index.map(is_holiday_week)

    # drop rows with NaN (from lag creation)
    before = len(df)
    df = df.dropna()
    logger.debug(f"Dropped {before - len(df)} rows due to NaN from lag features")

    return df


def get_feature_columns() -> list:
    """Return the list of feature cols — used consistently across train and predict."""
    return [
        "lag_1", "lag_4", "lag_8", "lag_52",
        "rolling_mean_4", "rolling_mean_8",
        "rolling_std_4", "rolling_std_8",
        "month", "quarter", "week_of_year",
        "day_of_week", "year", "is_q4", "holiday_flag",
    ]


def build_future_features(
    series: pd.Series, horizon: int = 8, label: str = "sales"
) -> pd.DataFrame:
    """
    Build features for future (unseen) weeks.
    We extend the series with NaN placeholders and fill iteratively.
    This mimics what happens at inference time.
    """
    # future dates
    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W"
    )

    # extend series with NaN for future
    future_series = pd.Series(np.nan, index=future_dates)
    extended = pd.concat([series, future_series])

    feat_df = build_features(extended, label=label)

    # return only the future rows (where target is NaN)
    future_feats = feat_df[feat_df.index.isin(future_dates)]
    return future_feats


if __name__ == "__main__":
    # quick test
    import pandas as pd
    import numpy as np

    idx = pd.date_range("2020-01-05", periods=100, freq="W")
    s = pd.Series(np.random.randint(1000, 5000, size=100), index=idx)
    df = build_features(s)
    print(df.tail())
    print(df.columns.tolist())
