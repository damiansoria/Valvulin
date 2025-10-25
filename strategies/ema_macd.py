"""Strategy combining EMA crosses with MACD confirmation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BaseStrategy
from .utils import ensure_records, exponential_moving_average


class EMAMACDStrategy(BaseStrategy):
    """Trade EMA crossovers confirmed by MACD direction."""

    default_params = {
        "ema_short": 12,
        "ema_long": 26,
        "macd_signal": 9,
    }

    def generate_signal(self, data: Sequence[Mapping[str, Any]] | Any) -> str | None:
        records = ensure_records(data)
        if len(records) < 3:
            return None

        ema_short_span = int(self.params["ema_short"])
        ema_long_span = int(self.params["ema_long"])
        signal_span = int(self.params["macd_signal"])

        closes = [float(row["close"]) for row in records]
        ema_short = exponential_moving_average(closes, ema_short_span)
        ema_long = exponential_moving_average(closes, ema_long_span)
        macd_line = [s - l for s, l in zip(ema_short, ema_long)]
        signal_line = exponential_moving_average(macd_line, signal_span)

        prev_diff = ema_short[-2] - ema_long[-2]
        curr_diff = ema_short[-1] - ema_long[-1]
        prev_macd = macd_line[-2] - signal_line[-2]
        curr_macd = macd_line[-1] - signal_line[-1]

        crossed_up = prev_diff <= 0 and curr_diff > 0 and curr_macd > 0
        crossed_down = prev_diff >= 0 and curr_diff < 0 and curr_macd < 0

        if crossed_up:
            return "buy"
        if crossed_down:
            return "sell"
        return None
