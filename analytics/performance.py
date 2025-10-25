"""Performance analytics helpers for el bot de trading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .trade_logger import TradeEntry


@dataclass
class PerformanceMetrics:
    """Container summarising key performance indicators.

    Attributes
    ----------
    win_rate:
        Ratio de operaciones ganadoras sobre el total (`0.0-1.0`).
    profit_factor:
        Relación entre beneficios brutos y pérdidas brutas.
    average_r_multiple:
        Promedio de `r_multiple` obtenido por operación.
    max_drawdown:
        Pérdida máxima acumulada durante la serie (en unidades de R).
    expectancy:
        Expectativa matemática usando la fórmula clásica de trading.
    total_trades:
        Número de operaciones consideradas para el cálculo.
    """

    win_rate: float
    profit_factor: float
    average_r_multiple: float
    max_drawdown: float
    expectancy: float
    total_trades: int

    def to_dict(self) -> dict:
        """Return the metrics as a serialisable mapping."""

        return {
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "average_r_multiple": self.average_r_multiple,
            "max_drawdown": self.max_drawdown,
            "expectancy": self.expectancy,
            "total_trades": self.total_trades,
        }

    @property
    def win_rate_percent(self) -> float:
        """Win rate expresado en porcentaje."""

        return self.win_rate * 100


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    """Compute the maximum drawdown (absolute units) from an equity curve."""

    peak = float("-inf")
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = peak - value
        max_dd = max(max_dd, drawdown)
    return max_dd


def _profit_factor(pnls: Sequence[float]) -> float:
    """Return the profit factor given a sequence of trade results."""

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _expectancy(r_values: Sequence[float]) -> float:
    """Calculate expectancy using win/loss ratios and average R multiples."""

    if not r_values:
        return 0.0
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    total = len(r_values)
    win_rate = len(wins) / total
    loss_rate = len(losses) / total
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return win_rate * avg_win + loss_rate * avg_loss


def compute_performance_metrics(trades: Iterable[TradeEntry]) -> PerformanceMetrics:
    """Compute aggregated performance metrics for the supplied trades."""

    trade_list: List[TradeEntry] = list(trades)
    total_trades = len(trade_list)
    if total_trades == 0:
        return PerformanceMetrics(
            win_rate=0.0,
            profit_factor=0.0,
            average_r_multiple=0.0,
            max_drawdown=0.0,
            expectancy=0.0,
            total_trades=0,
        )

    r_values = [trade.r_multiple for trade in trade_list]
    pnls = [trade.r_multiple for trade in trade_list]

    wins = sum(1 for r in r_values if r > 0)
    win_rate = wins / total_trades

    cumulative = 0.0
    equity_curve = []
    for r in r_values:
        cumulative += r
        equity_curve.append(cumulative)

    average_r = float(np.mean(r_values)) if r_values else 0.0
    return PerformanceMetrics(
        win_rate=win_rate,
        profit_factor=_profit_factor(pnls),
        average_r_multiple=average_r,
        max_drawdown=_max_drawdown(equity_curve),
        expectancy=_expectancy(r_values),
        total_trades=total_trades,
    )


def group_metrics_by_strategy(
    trades: Iterable[TradeEntry],
) -> dict[str, PerformanceMetrics]:
    """Compute metrics for each strategy represented in the trade list."""

    buckets: dict[str, List[TradeEntry]] = {}
    for trade in trades:
        buckets.setdefault(trade.strategy, []).append(trade)
    return {
        name: compute_performance_metrics(bucket) for name, bucket in buckets.items()
    }


__all__ = [
    "PerformanceMetrics",
    "compute_performance_metrics",
    "group_metrics_by_strategy",
]
