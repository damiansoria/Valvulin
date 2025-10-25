"""Breakout strategy that requires confirmation via volume expansion."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class BreakoutVolumeStrategy(BaseStrategy):
    """Detect price breakouts that happen with a volume surge."""

    default_params: Dict[str, Any] = {
        "breakout_window": 20,
        "volume_multiplier": 1.5,
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

        window = min(len(frame) - 1, breakout_window)
        rolling_high = (
            frame["high"].rolling(window=window, min_periods=1).max().shift(1)
        )
        avg_volume = (
            frame["volume"].rolling(window=window, min_periods=1).mean().shift(1)
        )

        breakout_condition = frame["close"] > rolling_high
        volume_condition = frame["volume"] >= volume_multiplier * avg_volume
        combined = breakout_condition & volume_condition

        signals.loc[combined.fillna(False)] = 1

        return pd.DataFrame({"signal": signals})
