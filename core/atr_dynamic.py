"""Helpers for ATR based risk management."""
from __future__ import annotations

import logging
from dataclasses import dataclass
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DynamicLevels:
    """Container holding ATR based stop-loss and take-profit levels."""

    stop_loss: float
    take_profit: float
    stop_distance: float


def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Average True Range for the provided OHLC dataframe."""

    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr.rename(f"ATR_{period}")


def calculate_levels(price: float, atr_value: float, atr_mult_sl: float, atr_mult_tp: float, direction: int) -> DynamicLevels:
    """Return ATR anchored stop-loss and take-profit levels."""

    atr_value = max(atr_value, 1e-8)
    stop_distance = atr_value * max(atr_mult_sl, 0.1)
    take_distance = atr_value * max(atr_mult_tp, 0.1)

    if direction > 0:
        stop_loss = price - stop_distance
        take_profit = price + take_distance
    else:
        stop_loss = price + stop_distance
        take_profit = price - take_distance

    return DynamicLevels(stop_loss=stop_loss, take_profit=take_profit, stop_distance=stop_distance)


def position_size(capital: float, risk_percent: float, atr_value: float, atr_mult_sl: float) -> float:
    """Calculate the position size so that ATR based stop equals the desired risk."""

    atr_value = max(atr_value, 1e-8)
    risk_fraction = max(risk_percent, 0.01) / 100
    stop_distance = atr_value * max(atr_mult_sl, 0.1)
    if stop_distance <= 0:
        return 0.0
    size = (capital * risk_fraction) / stop_distance
    return float(max(size, 0.0))


__all__ = ["DynamicLevels", "compute_atr", "calculate_levels", "position_size"]
