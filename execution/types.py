"""Common execution data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Optional


OrderSide = Literal["BUY", "SELL"]
EntryType = Literal["market", "limit"]


class OrderStatus(str, Enum):
    """Simplified order status enumeration."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass(slots=True)
class TradeSignal:
    """Signal produced by a strategy to open a position."""

    symbol: str
    timestamp: datetime
    side: OrderSide
    quantity: float
    entry_type: EntryType = "market"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_delta: Optional[float] = None
    client_tag: Optional[str] = None


@dataclass(slots=True)
class TradeResult:
    """Result of executing a trade signal."""

    signal: TradeSignal
    entry_price: Optional[float]
    exit_price: Optional[float]
    status: OrderStatus
    order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    message: Optional[str] = None
    filled_quantity: float = 0.0
    extra: dict = field(default_factory=dict)
