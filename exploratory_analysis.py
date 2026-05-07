"""
Exploratory Data Analysis — sales_data.xlsx
Run this to get a feel for the data before training.

I usually do this in a notebook but converting to .py for easier
version control and reproducibility.

Usage:
    python notebooks/exploratory_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

from src.data_preprocessing import load_raw_data, clean_and_parse, aggregate_weekly
from src.config import OUTPUTS_DIR

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def run_eda():
    print("=" * 60)
    print("SALES FORECASTING — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --- load and clean ---
    raw = load_raw_data()
    df = clean_and_parse(raw)

    print(f"\n📊 Raw Data Overview")
    print(f"  Shape       : {raw.shape}")
    print(f"  Columns     : {raw.columns.tolist()}")
    print(f"  States      : {df['State'].nunique()}")
    print(f"  Date range  : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Total rows  : {len(df)}")
    print(f"  Missing vals: {df.isnull().sum().to_dict()}")

    print(f"\n📈 Sales Distribution")
    print(df["Total"].describe().apply(lambda x: f"{x:,.0f}"))

    # state-level summary
    state_summary = df.groupby("State")["Total"].agg(["sum", "mean", "count"])
    state_summary.columns = ["Total Sales", "Avg per Transaction", "Transactions"]
    state_summary = state_summary.sort_values("Total Sales", ascending=False)
    print(f"\n🗺️  Top 10 States by Total Sales:")
    print(state_summary.head(10).to_string())

    # --- aggregate to weekly ---
    state_data = aggregate_weekly(df)

    # data length check
    lengths = {s: len(series) for s, series in state_data.items()}
    print(f"\n📅 Weekly Data Summary")
    print(f"  Min weeks : {min(lengths.values())} ({min(lengths, key=lengths.get)})")
    print(f"  Max weeks : {max(lengths.values())} ({max(lengths, key=lengths.get)})")
    print(f"  Avg weeks : {np.mean(list(lengths.values())):.1f}")

    # --- visualizations ---
    print("\n🎨 Generating plots...")

    # 1. top 5 states time series
    top5 = state_summary.head(5).index.tolist()
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    for i, state in enumerate(top5):
        s = state_data[state]
        axes[i].plot(s.index, s.values / 1e6, linewidth=1.5, color="#2563EB")
        axes[i].fill_between(s.index, s.values / 1e6, alpha=0.1, color="#2563EB")
        axes[i].set_title(state, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Sales ($M)", fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    fig.suptitle("Weekly Sales — Top 5 States", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "eda_top5_states.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ eda_top5_states.png")

    # 2. total sales by state (bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    top_states = state_summary.head(20)
    colors = ["#2563EB" if i < 5 else "#93C5FD" for i in range(len(top_states))]
    ax.bar(top_states.index, top_states["Total Sales"] / 1e9, color=colors)
    ax.set_title("Total Sales by State (Top 20)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Sales ($B)")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "eda_sales_by_state.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ eda_sales_by_state.png")

    # 3. seasonality check — average sales by month
    all_series = []
    for state, s in state_data.items():
        temp = pd.DataFrame({"sales": s.values, "month": s.index.month}, index=s.index)
        all_series.append(temp)
    combined = pd.concat(all_series)
    monthly_avg = combined.groupby("month")["sales"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.bar(month_names, monthly_avg.values / 1e6, color="#2563EB", alpha=0.8)
    ax.set_title("Average Weekly Sales by Month (all states)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Weekly Sales ($M)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "eda_monthly_seasonality.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ eda_monthly_seasonality.png")

    print(f"\n✅ EDA complete. Plots saved to {OUTPUTS_DIR}/")
    return state_data


if __name__ == "__main__":
    run_eda()
