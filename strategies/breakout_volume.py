"""Breakout strategy that requires confirmation via volume expansion."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class BreakoutVolumeStrategy(BaseStrategy):
    """Detect price breakouts that happen with a volume surge."""

    default_params: Dict[str, Any] = {
        "breakout_window": 20,
        "volume_multiplier": 1.5,
        "trend_window": 50,
        "min_range_pct": 0.005,
    }

    def generate_signals(self, data: pd.DataFrame) -> SignalDataFrame:
        """Return buy signals when price breaks the recent high on strong volume."""

        frame = data.copy()
        frame = frame.sort_index()
        signals = pd.Series(0, index=frame.index, dtype=int)

        if len(frame) < 2:
            return pd.DataFrame({"signal": signals})

        breakout_window = int(self.params["breakout_window"])
        volume_multiplier = float(self.params["volume_multiplier"])
        trend_window = int(self.params.get("trend_window", 50))
        min_range_pct = float(self.params.get("min_range_pct", 0.005))

        high = frame.get("high", frame["close"])
        low = frame.get("low", frame["close"])
        volume = frame.get("volume", pd.Series(1, index=frame.index))

        window = min(len(frame) - 1, breakout_window)
        rolling_high = high.rolling(window=window, min_periods=1).max().shift(1)
        avg_volume = volume.rolling(window=window, min_periods=1).mean().shift(1)

        ema_trend = frame["close"].ewm(span=trend_window, adjust=False).mean()
        trend_positive = ema_trend > ema_trend.shift(1)

        breakout_condition = frame["close"] > rolling_high
        volume_condition = volume >= volume_multiplier * avg_volume
        range_condition = (high - low).abs() / low.replace(0, pd.NA)
        range_condition = range_condition.fillna(0) >= min_range_pct
        combined = (
            breakout_condition & volume_condition & trend_positive & range_condition
        )

        signals.loc[combined.fillna(False)] = 1

        return pd.DataFrame({"signal": signals})
