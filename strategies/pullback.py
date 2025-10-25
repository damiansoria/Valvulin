"""Pullback strategy based on EMA20/EMA50 alignment."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BaseStrategy
from .utils import ensure_records, exponential_moving_average


class PullbackEMAStrategy(BaseStrategy):
    """Buy when price resumes the trend after an EMA pullback."""

    default_params = {
        "short_window": 20,
        "long_window": 50,
    }

    def generate_signal(self, data: Sequence[Mapping[str, Any]] | Any) -> str | None:
        records = ensure_records(data)
        if len(records) < 3:
            return None

        short_window = int(self.params["short_window"])
        long_window = int(self.params["long_window"])

        closes = [float(row["close"]) for row in records]
        ema_short = exponential_moving_average(closes, short_window)
        ema_long = exponential_moving_average(closes, long_window)

        prev_close = closes[-2]
        curr_close = closes[-1]
        prev_ema_short = ema_short[-2]
        curr_ema_short = ema_short[-1]
        prev_ema_long = ema_long[-2]
        curr_ema_long = ema_long[-1]

        trend_up = curr_ema_short > curr_ema_long and prev_ema_short > prev_ema_long
        pulled_back = prev_close < prev_ema_short and prev_close > prev_ema_long
        resumed = curr_close > curr_ema_short

        if trend_up and pulled_back and resumed:
            return "buy"
        return None
