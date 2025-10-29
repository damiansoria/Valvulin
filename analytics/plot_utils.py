"""Reusable plotting utilities for the Streamlit analytics dashboard."""
from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(df_trades: pd.DataFrame, *, drawdown: Optional[pd.Series] = None) -> plt.Figure:
    """Create an equity curve figure.

    Parameters
    ----------
    df_trades:
        DataFrame containing at least ``timestamp`` and ``capital_final`` columns.
    drawdown:
        Optional drawdown series to plot on a secondary axis.
    """

    figure, ax = plt.subplots(figsize=(10, 4))
    if "timestamp" in df_trades.columns:
        x_values: Iterable = df_trades["timestamp"]
    else:
        x_values = range(len(df_trades))

    ax.plot(x_values, df_trades["capital_final"], label="Equity", linewidth=2, color="#1f77b4")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time" if "timestamp" in df_trades.columns else "Trade #")
    ax.set_ylabel("Capital ($)")
    ax.grid(alpha=0.3)

    if drawdown is not None:
        ax2 = ax.twinx()
        ax2.plot(x_values, drawdown, label="Drawdown %", color="#d62728", linewidth=1.5, linestyle="--")
        ax2.set_ylabel("Drawdown (%)")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
    else:
        ax.legend(loc="upper left")

    figure.tight_layout()
    return figure


def plot_drawdown_curve(df_trades: pd.DataFrame) -> plt.Figure:
    """Plot drawdown percentage over time."""

    figure, ax = plt.subplots(figsize=(10, 4))
    if "timestamp" in df_trades.columns:
        x_values: Iterable = df_trades["timestamp"]
    else:
        x_values = range(len(df_trades))

    ax.fill_between(x_values, df_trades["drawdown"], color="#ff7f0e", alpha=0.3)
    ax.plot(x_values, df_trades["drawdown"], color="#ff7f0e", linewidth=2)
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Time" if "timestamp" in df_trades.columns else "Trade #")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=1)
    figure.tight_layout()
    return figure


def plot_r_multiple_distribution(df_trades: pd.DataFrame, *, bins: int = 30) -> plt.Figure:
    """Plot the distribution of R-multiples as a histogram."""

    figure, ax = plt.subplots(figsize=(8, 4))
    r_values = df_trades["r_multiple"].dropna()
    ax.hist(r_values, bins=bins, color="skyblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", label="Break-even")
    if not r_values.empty:
        ax.axvline(r_values.mean(), color="green", linestyle="--", label="Mean R")
    ax.set_title("R-Multiple Distribution")
    ax.set_xlabel("R-Multiple")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_expectancy_bar(expectancy_value: float, *, ylim: tuple[float, float] = (-2.0, 2.0)) -> plt.Figure:
    """Return a bar chart visualising expectancy."""

    figure, ax = plt.subplots(figsize=(4, 4))
    color = "green" if expectancy_value >= 0 else "red"
    ax.bar(["Expectancy"], [expectancy_value], color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Expectancy (R)")
    ax.set_title("Expectancy Signal")
    figure.tight_layout()
    return figure


__all__ = [
    "plot_equity_curve",
    "plot_drawdown_curve",
    "plot_r_multiple_distribution",
    "plot_expectancy_bar",
]
