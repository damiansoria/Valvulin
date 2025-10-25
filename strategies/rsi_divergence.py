"""RSI divergence strategy."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BaseStrategy
from .utils import ensure_records


class RSIDivergenceStrategy(BaseStrategy):
    """Detect bullish or bearish divergences between price and RSI."""

    default_params = {
        "divergence_type": "bullish",  # or "bearish" or "both"
    }

    def generate_signal(self, data: Sequence[Mapping[str, Any]] | Any) -> str | None:
        records = ensure_records(data)
        if len(records) < 2:
            return None

        divergence_type = self.params["divergence_type"]

        prev = records[-2]
        curr = records[-1]

        price_higher_high = float(curr["close"]) > float(prev["close"])
        price_lower_low = float(curr["close"]) < float(prev["close"])
        rsi_higher = float(curr["rsi"]) > float(prev["rsi"])
        rsi_lower = float(curr["rsi"]) < float(prev["rsi"])

        bullish = price_lower_low and rsi_higher
        bearish = price_higher_high and rsi_lower

        if divergence_type == "bullish" and bullish:
            return "buy"
        if divergence_type == "bearish" and bearish:
            return "sell"
        if divergence_type == "both":
            if bullish:
                return "buy"
            if bearish:
                return "sell"
        return None
