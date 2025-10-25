"""Plotting helpers for trading performance analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from .performance import group_metrics_by_strategy
from .trade_logger import TradeEntry


def _ensure_output_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_equity_curve(
    trades: Iterable[TradeEntry],
    output_path: Path | str | None = None,
    show: bool = False,
):
    """Plot an equity curve using cumulative R multiples."""

    trade_list = list(trades)
    cumulative = []
    running = 0.0
    for trade in trade_list:
        running += trade.r_multiple
        cumulative.append(running)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative, marker="o")
    ax.set_title("Equity Curve (R multiples)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative R")
    ax.grid(True, linestyle="--", alpha=0.4)

    output_path = _ensure_output_path(output_path)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_r_distribution(
    trades: Iterable[TradeEntry],
    bins: int = 20,
    output_path: Path | str | None = None,
    show: bool = False,
):
    """Plot histogram of trade R multiples."""

    r_values = [trade.r_multiple for trade in trades]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(r_values, bins=bins, color="#2b8a3e", alpha=0.7, edgecolor="black")
    ax.set_title("Distribución de R multiples")
    ax.set_xlabel("R multiple")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path = _ensure_output_path(output_path)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_strategy_patterns(
    trades: Iterable[TradeEntry],
    output_path: Path | str | None = None,
    show: bool = False,
):
    """Plot expectancy by strategy to highlight behavioural patterns."""

    trade_list = list(trades)
    metrics = group_metrics_by_strategy(trade_list)
    if not metrics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No hay datos", ha="center", va="center")
        ax.axis("off")
    else:
        labels = list(metrics.keys())
        expectancy = [m.expectancy for m in metrics.values()]
        win_rates = [m.win_rate for m in metrics.values()]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, expectancy, color="#1c7ed6", alpha=0.8)
        ax.set_title("Expectancy por estrategia")
        ax.set_ylabel("Expectancy (R)")
        ax.set_xlabel("Estrategia")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(bottom=min(0, min(expectancy)))

        for bar, rate in zip(bars, win_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"Win Rate: {rate:.1%}",
                ha="center",
                va="bottom",
            )

    output_path = _ensure_output_path(output_path)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


__all__ = [
    "plot_equity_curve",
    "plot_r_distribution",
    "plot_strategy_patterns",
]
