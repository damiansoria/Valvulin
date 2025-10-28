"""Pullback strategy based on EMA20/EMA50 alignment."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class PullbackEMAStrategy(BaseStrategy):
    """Buy when price resumes the trend after an EMA pullback."""

    default_params: Dict[str, int] = {
        "short_window": 20,
        "long_window": 50,
        "confirmation_window": 100,
        "pullback_bars": 3,
        "proximity_pct": 0.003,
    }

    def generate_signals(self, data: pd.DataFrame) -> SignalDataFrame:
        """Return long signals when price resumes an EMA-defined uptrend."""

        frame = data.copy()
        frame = frame.sort_index()
        signals = pd.Series(0, index=frame.index, dtype=int)

        if len(frame) < 3:
            return pd.DataFrame({"signal": signals})

        short_window = int(self.params["short_window"])
        long_window = int(self.params["long_window"])
        confirmation_window = int(self.params.get("confirmation_window", 100))
        pullback_bars = int(self.params.get("pullback_bars", 3))
        proximity_pct = float(self.params.get("proximity_pct", 0.003))

        open_prices = frame.get("open", frame["close"])
        high = frame.get("high", frame["close"])
        low = frame.get("low", frame["close"])
        volume = frame.get("volume", pd.Series(1, index=frame.index))

        ema_short = frame["close"].ewm(span=short_window, adjust=False).mean()
        ema_long = frame["close"].ewm(span=long_window, adjust=False).mean()
        ema_confirm = frame["close"].ewm(span=confirmation_window, adjust=False).mean()

        trend_up = (
            (ema_short > ema_long)
            & (ema_short.shift(1) > ema_long.shift(1))
            & (ema_confirm > ema_confirm.shift(1))
        )

        proximity = frame["close"].shift(1) <= ema_short.shift(1) * (1 + proximity_pct)
        above_long = frame["close"].shift(1) > ema_long.shift(1)
        pullback = proximity & above_long

        negative_moves = (frame["close"].diff() < 0).astype(int)
        pullback_length = negative_moves.rolling(window=pullback_bars, min_periods=1).sum()

        volume_trend = volume.diff().rolling(window=pullback_bars, min_periods=1).mean()
        decreasing_volume = (volume_trend <= 0) | volume_trend.isna()

        bullish_candle = frame["close"] >= open_prices
        wide_range = (high - low).abs().replace(0, 1)
        body = (frame["close"] - open_prices).abs()
        candle_quality = (body / wide_range).fillna(0)
        strong_reversal = bullish_candle & (candle_quality > 0.55)
        strong_reversal = strong_reversal | (open_prices == frame["close"])

        prev_decreasing_volume = decreasing_volume.shift(1).astype("boolean").fillna(True)

        setup = (
            trend_up
            & pullback
            & (pullback_length.shift(1).fillna(0) >= 1)
            & prev_decreasing_volume.astype(bool)
            & strong_reversal
            & (frame["close"] > ema_short)
        )

        signals.loc[setup.fillna(False)] = 1

        return pd.DataFrame({"signal": signals})
