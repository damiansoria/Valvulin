"""Inside bar consolidation strategy."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class InsideBarStrategy(BaseStrategy):
    """Identify inside bar patterns to prepare for breakout trades."""

    default_params: Dict[str, float] = {
        "max_ratio": 0.6,
    }

    def generate_signals(self, data: pd.DataFrame) -> SignalDataFrame:
        """Return a neutral signal while flagging rows that form inside bars."""

        frame = data.copy()
        frame = frame.sort_index()
        signals = pd.Series(0, index=frame.index, dtype=int)

        if len(frame) < 2:
            return pd.DataFrame(
                {"signal": signals, "inside_bar": pd.Series(False, index=frame.index)}
            )

        previous = frame.shift(1)
        max_ratio = float(self.params["max_ratio"])

        prev_range = previous["high"] - previous["low"]
        curr_range = frame["high"] - frame["low"]
        inside = (frame["high"] <= previous["high"]) & (frame["low"] >= previous["low"])
        small_range = (prev_range > 0) & (curr_range / prev_range <= max_ratio)
        inside_bar = inside & small_range

        return pd.DataFrame({"signal": signals, "inside_bar": inside_bar.fillna(False)})

    def generate_signal(self, data: Any) -> str | None:
        """Return ``"watch"`` when the latest candle forms an inside bar."""

        frame = self.prepare_data(data)
        signals = self.generate_signals(frame)
        if bool(signals["inside_bar"].iloc[-1]):
            return "watch"
        return super().generate_signal(frame)
