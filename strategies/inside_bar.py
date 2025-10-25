"""Inside bar consolidation strategy."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BaseStrategy
from .utils import ensure_records


class InsideBarStrategy(BaseStrategy):
    """Identify inside bar patterns to prepare for breakout trades."""

    default_params = {
        "max_ratio": 0.6,
    }

    def generate_signal(self, data: Sequence[Mapping[str, Any]] | Any) -> str | None:
        records = ensure_records(data)
        if len(records) < 2:
            return None

        max_ratio = float(self.params["max_ratio"])
        prev = records[-2]
        curr = records[-1]

        prev_range = float(prev["high"]) - float(prev["low"])
        curr_range = float(curr["high"]) - float(curr["low"])

        inside = float(curr["high"]) <= float(prev["high"]) and float(curr["low"]) >= float(prev["low"])
        small_range = prev_range > 0 and curr_range / prev_range <= max_ratio

        if inside and small_range:
            return "watch"
        return None
