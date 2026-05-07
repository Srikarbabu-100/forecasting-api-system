"""
Visualization utilities.
All plots save to outputs/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config import OUTPUTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]


def plot_forecast_vs_actual(
    train: pd.Series,
    test: pd.Series,
    predictions: dict,   # {model_name: np.array}
    state: str,
    save: bool = True,
):
    """
    Plot actual train/test vs model predictions.
    predictions dict can have multiple models for comparison.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # plot training history (last 52 weeks to keep it readable)
    ax.plot(
        train.index[-52:], train.values[-52:],
        color="gray", linewidth=1.5, alpha=0.6, label="Training (last year)"
    )
    ax.plot(
        test.index, test.values,
        color="black", linewidth=2, label="Actual", zorder=5
    )

    for i, (model_name, preds) in enumerate(predictions.items()):
        color = COLORS[i % len(COLORS)]
        pred_dates = test.index[:len(preds)]
        ax.plot(pred_dates, preds, color=color, linewidth=1.8,
                linestyle="--", label=model_name, zorder=4)

    ax.set_title(f"{state} — Forecast vs Actual", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, f"{state}_forecast_vs_actual.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info(f"Saved: {path}")

    plt.close()


def plot_model_comparison(comparison_df: pd.DataFrame, state: str, save: bool = True):
    """Bar chart comparing RMSE, MAE across models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["RMSE", "MAE", "MAPE"]

    for ax, metric in zip(axes, metrics):
        vals = comparison_df[metric].values
        models = comparison_df["model"].values
        bars = ax.bar(models, vals, color=COLORS[:len(models)], alpha=0.85)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("")

        # label bars
        for bar, val in zip(bars, vals):
            if metric == "MAPE":
                label = f"{val:.1f}%"
            else:
                label = f"${val:,.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                label, ha="center", va="bottom", fontsize=9
            )

        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(f"{state} — Model Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, f"{state}_model_comparison.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info(f"Saved: {path}")

    plt.close()


def plot_feature_importance(fi_df: pd.DataFrame, state: str, save: bool = True):
    """Horizontal bar chart of XGBoost feature importances."""
    top_n = fi_df.head(12)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top_n["feature"][::-1], top_n["importance"][::-1], color="#2563EB", alpha=0.8)
    ax.set_title(f"{state} — XGBoost Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")

    for bar, val in zip(bars, top_n["importance"][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, f"{state}_feature_importance.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info(f"Saved: {path}")

    plt.close()


def plot_weekly_sales_overview(state_data: dict, states_to_plot: list = None, save: bool = True):
    """
    Quick overview plot of all states (or a subset).
    Useful for sanity checking the data.
    """
    if states_to_plot is None:
        states_to_plot = list(state_data.keys())[:8]  # first 8 by default

    n = len(states_to_plot)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows))
    axes = axes.flatten()

    for i, state in enumerate(states_to_plot):
        ax = axes[i]
        series = state_data[state]
        ax.plot(series.index, series.values, linewidth=1.2, color="#2563EB")
        ax.set_title(state, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    # hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Weekly Sales by State (overview)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, "states_overview.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info(f"Saved: {path}")

    plt.close()
