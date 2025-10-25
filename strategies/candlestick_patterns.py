"""Simple candlestick pattern recognition strategy."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class CandlestickPatternStrategy(BaseStrategy):
    """Detect basic bullish or bearish candlestick reversals."""

    default_params: Dict[str, str] = {"pattern": "bullish_engulfing"}

    def generate_signals(self, data: pd.DataFrame) -> SignalDataFrame:
        """Generate engulfing pattern signals for the provided OHLC data."""

        frame = data.copy()
        frame = frame.sort_index()
        signals = pd.Series(0, index=frame.index, dtype=int)

        if len(frame) < 2:
            return pd.DataFrame({"signal": signals})

        previous = frame.shift(1)
        pattern = str(self.params.get("pattern", "bullish_engulfing")).lower()

        if pattern == "bullish_engulfing":
            prev_bearish = previous["close"] < previous["open"]
            curr_bullish = frame["close"] > frame["open"]
            body_engulf = (frame["close"] >= previous["open"]) & (
                frame["open"] <= previous["close"]
            )
            bullish = prev_bearish & curr_bullish & body_engulf
            signals.loc[bullish.fillna(False)] = 1
        elif pattern == "bearish_engulfing":
            prev_bullish = previous["close"] > previous["open"]
            curr_bearish = frame["close"] < frame["open"]
            body_engulf = (frame["close"] <= previous["open"]) & (
                frame["open"] >= previous["close"]
            )
            bearish = prev_bullish & curr_bearish & body_engulf
            signals.loc[bearish.fillna(False)] = -1

        return pd.DataFrame({"signal": signals})
