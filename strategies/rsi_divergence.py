"""RSI divergence strategy."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class RSIDivergenceStrategy(BaseStrategy):
    """Detect bullish or bearish divergences between price and RSI."""

    default_params: Dict[str, float] = {
        "divergence_type": "bullish",  # or "bearish" or "both"
        "structure_window": 10,
        "ema_period": 50,
        "rsi_threshold": 30,
    }

    def generate_signals(self, data: pd.DataFrame) -> SignalDataFrame:
        """Return divergence signals using closing price and RSI comparisons."""

        frame = data.copy()
        frame = frame.sort_index()
        signals = pd.Series(0, index=frame.index, dtype=int)

        if len(frame) < 2:
            return pd.DataFrame({"signal": signals})

        previous = frame.shift(1)
        divergence_type = str(self.params.get("divergence_type", "bullish")).lower()
        structure_window = int(self.params.get("structure_window", 10))
        ema_period = int(self.params.get("ema_period", 50))
        rsi_threshold = float(self.params.get("rsi_threshold", 30))

        ema = frame["close"].ewm(span=ema_period, adjust=False).mean()
        open_prices = frame.get("open", frame["close"])
        high = frame.get("high", frame["close"])
        low = frame.get("low", frame["close"])
        prev_high = high.rolling(structure_window, min_periods=1).max().shift(1)
        prev_low = low.rolling(structure_window, min_periods=1).min().shift(1)

        price_higher_high = frame["close"] > previous["close"]
        price_lower_low = frame["close"] < previous["close"]
        rsi_higher = frame["rsi"] > previous["rsi"]
        rsi_lower = frame["rsi"] < previous["rsi"]

        insufficient_history = len(frame) <= structure_window

        bullish_div = (
            price_lower_low
            & rsi_higher
            & ((frame["close"] > prev_high.fillna(frame["close"])) | insufficient_history)
            & ((ema > ema.shift(1)) | insufficient_history)
            & (frame["rsi"] > rsi_threshold)
            & (frame["close"] >= open_prices)
        )

        bearish_div = (
            price_higher_high
            & rsi_lower
            & ((frame["close"] < prev_low.fillna(frame["close"])) | insufficient_history)
            & ((ema < ema.shift(1)) | insufficient_history)
            & (frame["rsi"] < 100 - rsi_threshold)
            & (frame["close"] <= open_prices)
        )

        if divergence_type in {"bullish", "both"}:
            signals.loc[bullish_div.fillna(False)] = 1
        if divergence_type in {"bearish", "both"}:
            signals.loc[bearish_div.fillna(False)] = -1

        return pd.DataFrame({"signal": signals})
