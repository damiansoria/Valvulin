"""RSI divergence strategy."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class RSIDivergenceStrategy(BaseStrategy):
    """Detect bullish or bearish divergences between price and RSI."""

    default_params: Dict[str, str] = {
        "divergence_type": "bullish",  # or "bearish" or "both"
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

        price_higher_high = frame["close"] > previous["close"]
        price_lower_low = frame["close"] < previous["close"]
        rsi_higher = frame["rsi"] > previous["rsi"]
        rsi_lower = frame["rsi"] < previous["rsi"]

        bullish = price_lower_low & rsi_higher
        bearish = price_higher_high & rsi_lower

        if divergence_type in {"bullish", "both"}:
            signals.loc[bullish.fillna(False)] = 1
        if divergence_type in {"bearish", "both"}:
            signals.loc[bearish.fillna(False)] = -1

        return pd.DataFrame({"signal": signals})
