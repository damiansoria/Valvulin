"""Utility helpers shared across strategy implementations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, List

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - if pandas is unavailable we fall back to sequences
    pd = None  # type: ignore


def ensure_records(data: Any) -> List[Mapping[str, Any]]:
    """Return the data as a list of mapping records."""

    if pd is not None and hasattr(data, "to_dict"):
        if isinstance(data, pd.DataFrame):  # type: ignore
            return [row for _, row in data.iterrows()]
    if isinstance(data, Sequence) and data and isinstance(data[0], Mapping):
        return list(data)  # type: ignore[return-value]
    raise TypeError("Data must be a pandas.DataFrame or sequence of mappings")


def get_column(data: Any, key: str) -> List[float]:
    records = ensure_records(data)
    return [float(record[key]) for record in records]


def get_last_record(data: Any) -> Mapping[str, Any]:
    records = ensure_records(data)
    return records[-1]


def get_prev_record(data: Any) -> Mapping[str, Any]:
    records = ensure_records(data)
    if len(records) < 2:
        raise ValueError("Need at least two records")
    return records[-2]


def exponential_moving_average(values: Sequence[float], span: int) -> List[float]:
    if span <= 0:
        raise ValueError("span must be positive")
    alpha = 2 / (span + 1)
    ema: List[float] = []
    for value in values:
        if not ema:
            ema.append(value)
        else:
            ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema
