"""Base class for trading strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStrategy(ABC):
    """Abstract base class for all strategies.

    Parameters are stored in ``self.params`` to make them discoverable and
    configurable for callers. Concrete strategies should document the expected
    columns in the input data passed to :meth:`generate_signal`.
    """

    default_params: Dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = {**self.default_params, **params}
        self.state: Dict[str, Any] = {}

    @abstractmethod
    def generate_signal(self, data: Any) -> Any:
        """Return the trading signal for the provided data."""

    def update_state(self, event: Dict[str, Any]) -> None:
        """Update mutable strategy state in response to market events."""

        self.state.update(event)
