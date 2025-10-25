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

        ema_short = frame["close"].ewm(span=short_window, adjust=False).mean()
        ema_long = frame["close"].ewm(span=long_window, adjust=False).mean()

        trend_up = (ema_short > ema_long) & (ema_short.shift(1) > ema_long.shift(1))
        pulled_back = (frame["close"].shift(1) < ema_short.shift(1)) & (
            frame["close"].shift(1) > ema_long.shift(1)
        )
        resumed = frame["close"] > ema_short
        setup = trend_up & pulled_back & resumed

        signals.loc[setup.fillna(False)] = 1

        return pd.DataFrame({"signal": signals})
