"""Simple candlestick pattern recognition strategy."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BaseStrategy
from .utils import ensure_records


class CandlestickPatternStrategy(BaseStrategy):
    """Detect basic bullish or bearish candlestick reversals."""

    default_params = {"pattern": "bullish_engulfing"}

    def generate_signal(self, data: Sequence[Mapping[str, Any]] | Any) -> str | None:
        records = ensure_records(data)
        if len(records) < 2:
            return None

        pattern = self.params["pattern"]
        prev = records[-2]
        curr = records[-1]

        if pattern == "bullish_engulfing":
            prev_bearish = float(prev["close"]) < float(prev["open"])
            curr_bullish = float(curr["close"]) > float(curr["open"])
            body_engulf = (
                float(curr["close"]) >= float(prev["open"])
                and float(curr["open"]) <= float(prev["close"])
            )
            if prev_bearish and curr_bullish and body_engulf:
                return "buy"
        elif pattern == "bearish_engulfing":
            prev_bullish = float(prev["close"]) > float(prev["open"])
            curr_bearish = float(curr["close"]) < float(curr["open"])
            body_engulf = (
                float(curr["close"]) <= float(prev["open"])
                and float(curr["open"]) >= float(prev["close"])
            )
            if prev_bullish and curr_bearish and body_engulf:
                return "sell"

        return None
