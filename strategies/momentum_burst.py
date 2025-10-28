"""Momentum burst strategy that reacts to explosive candles."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class MomentumBurstStrategy(BaseStrategy):
    """Detect high momentum bursts with exceptional range and volume."""

    default_params: Dict[str, float] = {
        "range_multiplier": 2.5,
        "volume_multiplier": 2.0,
        "atr_window": 14,
        "trend_window": 50,
    }

    def generate_signals(self, data: pd.DataFrame) -> SignalDataFrame:
        frame = data.copy().sort_index()
        signals = pd.Series(0, index=frame.index, dtype=int)

        if len(frame) < 5:
            return pd.DataFrame({"signal": signals})

        atr_window = int(self.params.get("atr_window", 14))
        range_multiplier = float(self.params.get("range_multiplier", 2.5))
        volume_multiplier = float(self.params.get("volume_multiplier", 2.0))
        trend_window = int(self.params.get("trend_window", 50))

        true_range = (frame["high"] - frame["low"]).abs()
        avg_range = true_range.rolling(window=atr_window, min_periods=1).mean()
        avg_volume = frame["volume"].rolling(window=atr_window, min_periods=1).mean()
        ema_trend = frame["close"].ewm(span=trend_window, adjust=False).mean()
        trend_positive = ema_trend > ema_trend.shift(1)

        range_condition = true_range >= avg_range * range_multiplier
        volume_condition = frame["volume"] >= avg_volume * volume_multiplier
        strong_close = frame["close"] >= frame["high"] - (true_range * 0.1)

        long_signal = range_condition & volume_condition & strong_close & trend_positive

        signals.loc[long_signal.fillna(False)] = 1

        return pd.DataFrame({"signal": signals})
