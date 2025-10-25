"""Performance analytics helpers for the trading bot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .trade_logger import TradeEntry


@dataclass
class PerformanceMetrics:
    """Container summarising key performance indicators."""

    win_rate: float
    profit_factor: float
    max_drawdown: float
    expectancy: float
    total_trades: int
    total_r_multiple: float

    def to_dict(self) -> dict:
        return {
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "expectancy": self.expectancy,
            "total_trades": self.total_trades,
            "total_r_multiple": self.total_r_multiple,
        }


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = peak - value
        max_dd = max(max_dd, drawdown)
    return max_dd


def compute_performance_metrics(trades: Iterable[TradeEntry]) -> PerformanceMetrics:
    """Compute aggregated performance metrics for the supplied trades."""

    trade_list: List[TradeEntry] = list(trades)
    total_trades = len(trade_list)
    if total_trades == 0:
        return PerformanceMetrics(
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            expectancy=0.0,
            total_trades=0,
            total_r_multiple=0.0,
        )

    r_values = [trade.r_multiple for trade in trade_list]
    wins = sum(1 for r in r_values if r > 0)
    losses = [r for r in r_values if r < 0]
    profits = [r for r in r_values if r > 0]

    win_rate = wins / total_trades

    gross_profit = sum(profits)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity_curve = []
    cumulative = 0.0
    for r in r_values:
        cumulative += r
        equity_curve.append(cumulative)
    max_drawdown = _max_drawdown(equity_curve)

    expectancy = sum(r_values) / total_trades
    total_r_multiple = sum(r_values)

    return PerformanceMetrics(
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        expectancy=expectancy,
        total_trades=total_trades,
        total_r_multiple=total_r_multiple,
    )


def group_metrics_by_strategy(trades: Iterable[TradeEntry]) -> dict[str, PerformanceMetrics]:
    """Compute metrics for each strategy represented in the trade list."""

    buckets: dict[str, List[TradeEntry]] = {}
    for trade in trades:
        buckets.setdefault(trade.strategy, []).append(trade)
    return {name: compute_performance_metrics(bucket) for name, bucket in buckets.items()}


__all__ = [
    "PerformanceMetrics",
    "compute_performance_metrics",
    "group_metrics_by_strategy",
]
