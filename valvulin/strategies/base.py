"""Strategy abstractions for trading and backtesting."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd

from valvulin.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Signal:
    """Represents a trading signal emitted by a strategy."""

    symbol: str
    action: str  # e.g. "buy", "sell", "hold"
    confidence: float
    metadata: Dict[str, float]


class Strategy(ABC):
    """Base strategy class implementing the core lifecycle methods."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def prepare(self, data: pd.DataFrame) -> None:
        """Called once to allow the strategy to warm up indicators."""

    @abstractmethod
    def generate(self, data: pd.DataFrame) -> Iterable[Signal]:
        """Produce signals for the provided slice of data."""

    def log_signal(self, signal: Signal) -> None:
        """Log structured signal information for auditing."""

        logger.info(
            "strategy_signal",
            extra={
                "strategy": self.name,
                "symbol": signal.symbol,
                "action": signal.action,
                "confidence": signal.confidence,
                **signal.metadata,
            },
        )


__all__ = ["Strategy", "Signal"]
