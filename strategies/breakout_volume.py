"""Breakout strategy that requires confirmation via volume expansion."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BaseStrategy
from .utils import ensure_records


class BreakoutVolumeStrategy(BaseStrategy):
    """Detect price breakouts that happen with a volume surge."""

    default_params = {
        "breakout_window": 20,
        "volume_multiplier": 1.5,
    }

    def generate_signal(self, data: Sequence[Mapping[str, Any]] | Any) -> str | None:
        records = ensure_records(data)
        if len(records) < 2:
            return None

        breakout_window = int(self.params["breakout_window"])
        volume_multiplier = float(self.params["volume_multiplier"])

        window = min(len(records) - 1, breakout_window)
        recent = records[-(window + 1) :]
        history = recent[:-1]
        current = recent[-1]

        highs = [float(row["high"]) for row in history]
        volumes = [float(row["volume"]) for row in history]

        highest_high = max(highs)
        avg_volume = sum(volumes) / len(volumes)

        is_breakout = float(current["close"]) > highest_high
        volume_ok = float(current["volume"]) >= volume_multiplier * avg_volume

        if is_breakout and volume_ok:
            return "buy"
        return None
