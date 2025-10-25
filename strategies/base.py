"""Core abstractions shared by all standalone strategies.

This module defines :class:`BaseStrategy`, a lightweight base class used by the
strategies bundled with the project.  Unlike the asynchronous strategies used
by the live engine, these helpers operate on pandas ``DataFrame`` instances and
return tabular signals suitable for backtesting or offline analytics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable

import pandas as pd


SignalDataFrame = pd.DataFrame


class BaseStrategy(ABC):
    """Base behaviour shared by the simple, DataFrame-based strategies.

    Parameters supplied to the constructor are merged with
    :attr:`default_params` and stored in :attr:`self.params`.  Concrete
    implementations operate on OHLCV style data and **must** implement
    :meth:`generate_signals`, returning a :class:`pandas.DataFrame` that
    contains a ``signal`` column with values ``1`` (buy), ``-1`` (sell) or ``0``
    (neutral).
    """

    default_params: Dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = {**self.default_params, **params}
        self.state: Dict[str, Any] = {}

    def prepare_data(self, data: Any) -> pd.DataFrame:
        """Return the supplied ``data`` as a clean :class:`pandas.DataFrame`.

        Parameters
        ----------
        data:
            A :class:`pandas.DataFrame` or an iterable of mapping objects.

        Returns
        -------
        pandas.DataFrame
            Data indexed identically to the input (when possible) and ordered
            chronologically.
        """

        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            frame = pd.DataFrame(list(data))
        else:
            raise TypeError(
                "Data must be a pandas.DataFrame or an iterable of mapping objects"
            )

        if frame.empty:
            raise ValueError("Strategy data must contain at least one row")

        if not isinstance(frame.index, (pd.DatetimeIndex, pd.RangeIndex)):
            frame.index = pd.RangeIndex(start=0, stop=len(frame))

        return frame

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> SignalDataFrame:
        """Return a DataFrame with a ``signal`` column populated with ``-1``/``0``/``1``."""

    def generate_signal(self, data: Any) -> str | None:
        """Legacy helper that returns the most recent textual signal.

        The helper exists for backwards compatibility with downstream code and
        unit tests that relied on a string output.  ``"buy"`` is returned for
        positive signals, ``"sell"`` for negative signals and ``None`` when the
        last value is neutral.
        """

        frame = self.prepare_data(data)
        signals = self.generate_signals(frame)
        if "signal" not in signals.columns:
            raise ValueError("Strategy output must contain a 'signal' column")
        last_value = int(signals["signal"].iloc[-1])
        if last_value > 0:
            return "buy"
        if last_value < 0:
            return "sell"
        return None

    def update_state(self, event: Dict[str, Any]) -> None:
        """Update mutable strategy state in response to market events."""

        self.state.update(event)
