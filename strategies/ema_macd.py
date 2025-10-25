"""Strategy combining EMA crosses with MACD confirmation."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class EMAMACDStrategy(BaseStrategy):
    """Trade EMA crossovers confirmed by MACD direction."""

    default_params: Dict[str, int] = {
        "ema_short": 12,
        "ema_long": 26,
        "macd_signal": 9,
    }

    def generate_signals(self, data: pd.DataFrame) -> SignalDataFrame:
        """Return long/short signals using EMA crossovers and MACD confirmation."""

        frame = data.copy()
        frame = frame.sort_index()
        signals = pd.Series(0, index=frame.index, dtype=int)

        if len(frame) < 3:
            return pd.DataFrame({"signal": signals})

        ema_short_span = int(self.params["ema_short"])
        ema_long_span = int(self.params["ema_long"])
        signal_span = int(self.params["macd_signal"])

        ema_short = frame["close"].ewm(span=ema_short_span, adjust=False).mean()
        ema_long = frame["close"].ewm(span=ema_long_span, adjust=False).mean()
        ema_diff = ema_short - ema_long
        macd_line = ema_diff
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()

        ema_diff_prev = ema_diff.shift(1)

        crossed_up = (ema_diff_prev <= 0) & (ema_diff > 0) & (macd_line > signal_line)
        crossed_down = (ema_diff_prev >= 0) & (ema_diff < 0) & (macd_line < signal_line)

        signals.loc[crossed_up.fillna(False)] = 1
        signals.loc[crossed_down.fillna(False)] = -1

        return pd.DataFrame({"signal": signals})
