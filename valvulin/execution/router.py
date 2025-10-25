"""Order execution interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from valvulin.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Order:
    """Represents an order to be sent to an exchange."""

    symbol: str
    side: str
    quantity: float
    price: float | None = None
    order_type: str = "MARKET"


class ExchangeClient(Protocol):
    """Protocol describing exchange clients used for live trading."""

    def create_order(self, *, symbol: str, side: str, type: str, quantity: float, price: float | None = None) -> dict: ...


class OrderRouter:
    """Routes orders through an exchange client while logging activity."""

    def __init__(self, client: ExchangeClient) -> None:
        self.client = client

    def send(self, order: Order) -> dict:
        logger.info(
            "execution_send_order",
            extra={
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "order_type": order.order_type,
            },
        )
        payload = {
            "symbol": order.symbol,
            "side": order.side.upper(),
            "type": order.order_type.upper(),
            "quantity": order.quantity,
        }
        if order.price is not None:
            payload["price"] = order.price
        response = self.client.create_order(**payload)
        logger.info("execution_order_response", extra={"response": response})
        return response


__all__ = ["Order", "OrderRouter", "ExchangeClient"]
