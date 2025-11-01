"""Trend and volatility filters used before opening trades."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FilterContext:
    """Pre-computed indicators used to filter trade entries."""

    adx: pd.Series
    atr: pd.Series
    atr_threshold: pd.Series
    strategy_columns: Dict[str, str]


def compute_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Average Directional Index for the dataframe."""

    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)

    up_move = high.diff()
    down_move = low.diff() * -1

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    atr = true_range.rolling(window=period, min_periods=1).mean()

    plus_di = 100 * pd.Series(plus_dm).rolling(period, min_periods=1).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).rolling(period, min_periods=1).mean() / atr.replace(0, np.nan)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(window=period, min_periods=1).mean().fillna(0.0)
    return pd.Series(adx, index=data.index, name=f"ADX_{period}")


def build_context(
    *,
    data: pd.DataFrame,
    adx_period: int = 14,
    atr_period: int = 14,
    atr_quantile: float = 0.6,
    strategy_columns: Dict[str, str] | None = None,
) -> FilterContext:
    """Create a :class:`FilterContext` ready to be used while iterating the dataframe."""

    adx = compute_adx(data, period=adx_period)
    atr = data.get(f"ATR_{atr_period}")
    if atr is None:
        raise ValueError("ATR column must be computed before building the filter context.")
    threshold = atr.rolling(window=atr_period * 2, min_periods=atr_period).quantile(atr_quantile)
    threshold = threshold.combine_first(atr)
    threshold = threshold.bfill().ffill().fillna(atr.mean())
    return FilterContext(adx=adx, atr=atr, atr_threshold=threshold, strategy_columns=strategy_columns or {})


def allow_entry(
    *,
    context: FilterContext,
    index: int,
    signal: int,
    row: pd.Series,
) -> bool:
    """Evaluate whether a trade can be opened given the current context."""

    if signal == 0:
        return False

    adx_value = float(context.adx.iloc[index]) if len(context.adx) > index else 0.0
    atr_value = float(context.atr.iloc[index]) if len(context.atr) > index else 0.0
    threshold = float(context.atr_threshold.iloc[index]) if len(context.atr_threshold) > index else 0.0

    if adx_value <= 20 and atr_value <= threshold:
        if threshold <= 0 and atr_value <= 0:
            pass
        else:
            return False

    rsi_col = context.strategy_columns.get("RSI")
    sma_col = context.strategy_columns.get("SMA Crossover")

    if rsi_col and rsi_col in row.index:
        if int(row[rsi_col]) != signal:
            return False
    if sma_col and sma_col in row.index:
        if int(row[sma_col]) != signal:
            return False

    return True


__all__ = ["FilterContext", "build_context", "allow_entry", "compute_adx"]
