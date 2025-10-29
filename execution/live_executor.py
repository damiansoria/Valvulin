"""Live execution helper for Binance trading."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import requests

from .types import OrderStatus, TradeResult, TradeSignal

try:  # pragma: no cover - optional dependency
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceOrderException
except Exception:  # pragma: no cover - when python-binance is not installed
    Client = None

    class BinanceAPIException(Exception):
        """Fallback Binance API exception."""

    class BinanceOrderException(Exception):
        """Fallback Binance order exception."""


SIDE_BUY = getattr(Client, "SIDE_BUY", "BUY")
SIDE_SELL = getattr(Client, "SIDE_SELL", "SELL")
ORDER_TYPE_MARKET = getattr(Client, "ORDER_TYPE_MARKET", "MARKET")
ORDER_TYPE_LIMIT = getattr(Client, "ORDER_TYPE_LIMIT", "LIMIT")
ORDER_TYPE_STOP_LOSS_LIMIT = getattr(Client, "ORDER_TYPE_STOP_LOSS_LIMIT", "STOP_LOSS_LIMIT")
ORDER_TYPE_TAKE_PROFIT_LIMIT = getattr(Client, "ORDER_TYPE_TAKE_PROFIT_LIMIT", "TAKE_PROFIT_LIMIT")
TIME_IN_FORCE_GTC = getattr(Client, "TIME_IN_FORCE_GTC", "GTC")


class ExecutionError(RuntimeError):
    """Generic execution error."""


@dataclass(slots=True)
class PaperOrder:
    """Represents a simulated order."""

    signal: TradeSignal
    order_id: str
    entry_price: Optional[float]
    filled: bool = False
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0


@dataclass(slots=True)
class PaperPosition:
    """Tracks an open simulated position."""

    signal: TradeSignal
    entry_price: float
    quantity: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    order_id: str
    created_at: float = field(default_factory=time.time)


class PaperBroker:
    """Lightweight paper trading broker."""

    def __init__(self, price_getter: Callable[[str], float]) -> None:
        if price_getter is None:
            raise ValueError("Paper trading requires a price getter callable")
        self.price_getter = price_getter
        self.orders: Dict[str, PaperOrder] = {}
        self.positions: Dict[str, PaperPosition] = {}
        self._order_seq = 0

    def execute(self, signal: TradeSignal) -> TradeResult:
        order_id = self._next_order_id()
        current_price = self.price_getter(signal.symbol)
        if current_price is None:
            raise ExecutionError(f"No market price available for {signal.symbol}")

        should_fill = False
        fill_price = None
        if signal.entry_type == "market":
            should_fill = True
            fill_price = current_price
        else:
            target = signal.entry_price
            if target is None:
                raise ExecutionError("Limit order without entry price")
            if signal.side == "BUY" and current_price <= target:
                should_fill = True
                fill_price = target
            elif signal.side == "SELL" and current_price >= target:
                should_fill = True
                fill_price = target

        order = PaperOrder(signal=signal, order_id=order_id, entry_price=fill_price)
        self.orders[order_id] = order

        if should_fill and fill_price is not None:
            order.filled = True
            order.status = OrderStatus.FILLED
            order.filled_quantity = signal.quantity
            position = PaperPosition(
                signal=signal,
                entry_price=fill_price,
                quantity=signal.quantity,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                order_id=order_id,
            )
            self.positions[order_id] = position
        else:
            order.status = OrderStatus.NEW

        return TradeResult(
            signal=signal,
            entry_price=fill_price if should_fill else None,
            exit_price=None,
            status=order.status,
            order_id=order_id,
            filled_quantity=order.filled_quantity,
        )

    def process_price(self, symbol: str, price: float) -> List[TradeResult]:
        closed: List[TradeResult] = []
        # Fill pending limit orders
        for order in list(self.orders.values()):
            if order.signal.symbol != symbol or order.filled:
                continue
            target = order.signal.entry_price
            if target is None:
                continue
            if order.signal.side == "BUY" and price <= target:
                self._fill_order(order, target)
            elif order.signal.side == "SELL" and price >= target:
                self._fill_order(order, target)

        # Evaluate exits for open positions
        for position in list(self.positions.values()):
            if position.signal.symbol != symbol:
                continue
            exit_price, status = self._evaluate_exit(position, price)
            if exit_price is not None:
                closed.append(self._close_position(position, exit_price, status))
        return closed

    def _fill_order(self, order: PaperOrder, fill_price: float) -> None:
        if order.filled:
            return
        order.filled = True
        order.status = OrderStatus.FILLED
        order.entry_price = fill_price
        order.filled_quantity = order.signal.quantity
        position = PaperPosition(
            signal=order.signal,
            entry_price=fill_price,
            quantity=order.signal.quantity,
            stop_loss=order.signal.stop_loss,
                take_profit=order.signal.take_profit,
            order_id=order.order_id,
        )
        self.positions[order.order_id] = position

    def _evaluate_exit(self, position: PaperPosition, price: float) -> tuple[Optional[float], OrderStatus]:
        signal = position.signal
        exit_price = None
        status = OrderStatus.FILLED

        if signal.side == "BUY":
            if position.stop_loss is not None and price <= position.stop_loss:
                exit_price = position.stop_loss
            elif position.take_profit is not None and price >= position.take_profit:
                exit_price = position.take_profit
        else:
            if position.stop_loss is not None and price >= position.stop_loss:
                exit_price = position.stop_loss
            elif position.take_profit is not None and price <= position.take_profit:
                exit_price = position.take_profit

        return exit_price, status

    def _close_position(self, position: PaperPosition, exit_price: float, status: OrderStatus) -> TradeResult:
        order = self.orders[position.order_id]
        pnl = self._compute_pnl(position.signal.side, position.entry_price, exit_price, position.quantity)
        self.positions.pop(position.order_id, None)
        order.status = status
        return TradeResult(
            signal=position.signal,
            entry_price=position.entry_price,
            exit_price=exit_price,
            status=status,
            order_id=position.order_id,
            filled_quantity=position.quantity,
            extra={"pnl": pnl},
        )

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"paper-{self._order_seq}"

    @staticmethod
    def _compute_pnl(side: str, entry_price: float, exit_price: float, quantity: float) -> float:
        direction = 1 if side == "BUY" else -1
        return direction * (exit_price - entry_price) * quantity


class LiveExecutor:
    """Transform trade signals into Binance orders."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        testnet: bool = False,
        paper_trading: bool = False,
        price_getter: Optional[Callable[[str], float]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.paper_trading = paper_trading
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._recoverable_errors = (
            BinanceAPIException,
            BinanceOrderException,
            requests.RequestException,
        )

        if paper_trading:
            if price_getter is None:
                raise ExecutionError("paper_trading requires a price_getter callable for live pricing")
            self.client = None
            self.broker = PaperBroker(price_getter)
        else:
            if Client is None:
                raise ExecutionError("python-binance is required for live execution")
            self.client = Client(api_key, api_secret, testnet=testnet)
            self.broker = None

    def execute_signal(self, signal: TradeSignal, wait_fill: bool = True) -> TradeResult:
        if self.paper_trading:
            return self.broker.execute(signal)

        order = self._place_entry_order(signal)
        order_id = str(order.get("orderId"))
        status = order.get("status", "NEW")
        entry_price = self._extract_filled_price(order)

        if wait_fill and status != OrderStatus.FILLED.value:
            order = self._await_fill(signal.symbol, order_id)
            status = order.get("status", status)
            entry_price = self._extract_filled_price(order)

        try:
            result_status = OrderStatus(status)
        except ValueError:
            result_status = OrderStatus.NEW

        stop_order_id: Optional[str] = None
        take_profit_order_id: Optional[str] = None

        if result_status == OrderStatus.FILLED:
            stop_order_id, take_profit_order_id = self._place_brackets(signal, entry_price)

        return TradeResult(
            signal=signal,
            entry_price=entry_price,
            exit_price=None,
            status=result_status,
            order_id=order_id,
            stop_order_id=stop_order_id,
            take_profit_order_id=take_profit_order_id,
            filled_quantity=float(order.get("executedQty", 0.0)),
            extra={"raw": order},
        )

    def sync_open_orders(self, symbol: str) -> List[dict]:
        if self.paper_trading:
            return [
                {
                    "orderId": order.order_id,
                    "status": order.status.value,
                    "symbol": order.signal.symbol,
                    "price": order.entry_price,
                    "origQty": order.signal.quantity,
                    "executedQty": order.filled_quantity,
                }
                for order in self.broker.orders.values()
                if order.signal.symbol == symbol
            ]
        return self._call_with_retries(self.client.get_open_orders, symbol=symbol)

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        if self.paper_trading:
            order = self.broker.orders.get(order_id)
            if order is None:
                raise ExecutionError(f"Unknown order {order_id}")
            order.status = OrderStatus.CANCELED
            self.broker.positions.pop(order_id, None)
            return {"orderId": order_id, "status": OrderStatus.CANCELED.value}
        return self._call_with_retries(self.client.cancel_order, symbol=symbol, orderId=order_id)

    def cancel_all(self, symbol: str) -> None:
        open_orders = self.sync_open_orders(symbol)
        for order in open_orders:
            try:
                self.cancel_order(symbol, str(order["orderId"]))
            except ExecutionError:
                continue

    def process_price_update(self, symbol: str, price: float) -> List[TradeResult]:
        if not self.paper_trading:
            raise ExecutionError("process_price_update is only available in paper trading mode")
        return self.broker.process_price(symbol, price)

    def _place_entry_order(self, signal: TradeSignal) -> dict:
        params = {
            "symbol": signal.symbol,
            "side": signal.side,
            "quantity": self._format_quantity(signal.quantity),
            "type": ORDER_TYPE_MARKET if signal.entry_type == "market" else ORDER_TYPE_LIMIT,
        }
        if signal.entry_type == "limit":
            if signal.entry_price is None:
                raise ExecutionError("Limit orders require entry_price")
            params.update({"price": self._format_price(signal.entry_price), "timeInForce": TIME_IN_FORCE_GTC})
        return self._call_with_retries(self.client.create_order, **params)

    def _place_brackets(self, signal: TradeSignal, entry_price: Optional[float]) -> tuple[Optional[str], Optional[str]]:
        closing_side = SIDE_SELL if signal.side == SIDE_BUY else SIDE_BUY
        stop_id: Optional[str] = None
        take_id: Optional[str] = None

        if signal.stop_loss is not None and signal.take_profit is not None:
            stop_limit_price = self._offset_price(signal.stop_loss, -0.001 if signal.side == SIDE_BUY else 0.001)
            response = self._call_with_retries(
                self.client.create_oco_order,
                symbol=signal.symbol,
                side=closing_side,
                quantity=self._format_quantity(signal.quantity),
                price=self._format_price(signal.take_profit),
                stopPrice=self._format_price(signal.stop_loss),
                stopLimitPrice=self._format_price(stop_limit_price),
                stopLimitTimeInForce=TIME_IN_FORCE_GTC,
            )
            stop_id = str(response["orderReports"][1]["orderId"])
            take_id = str(response["orderReports"][0]["orderId"])
        else:
            if signal.take_profit is not None:
                response = self._call_with_retries(
                    self.client.create_order,
                    symbol=signal.symbol,
                    side=closing_side,
                    type=ORDER_TYPE_TAKE_PROFIT_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    price=self._format_price(signal.take_profit),
                    stopPrice=self._format_price(signal.take_profit),
                    quantity=self._format_quantity(signal.quantity),
                )
                take_id = str(response.get("orderId"))
            if signal.stop_loss is not None:
                stop_limit_price = self._offset_price(signal.stop_loss, -0.001 if signal.side == SIDE_BUY else 0.001)
                response = self._call_with_retries(
                    self.client.create_order,
                    symbol=signal.symbol,
                    side=closing_side,
                    type=ORDER_TYPE_STOP_LOSS_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    price=self._format_price(stop_limit_price),
                    stopPrice=self._format_price(signal.stop_loss),
                    quantity=self._format_quantity(signal.quantity),
                )
                stop_id = str(response.get("orderId"))

        return stop_id, take_id

    def _await_fill(self, symbol: str, order_id: str) -> dict:
        for attempt in range(self.max_retries):
            order = self._call_with_retries(self.client.get_order, symbol=symbol, orderId=order_id)
            status = order.get("status")
            if status == OrderStatus.FILLED.value:
                return order
            time.sleep(self.retry_delay * (attempt + 1))
        raise ExecutionError(f"Order {order_id} not filled after retries")

    def _call_with_retries(self, func: Callable, *args, **kwargs):
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except self._recoverable_errors as exc:
                if attempt >= self.max_retries:
                    raise ExecutionError(str(exc)) from exc
                time.sleep(self.retry_delay * attempt)

    @staticmethod
    def _format_quantity(quantity: float) -> str:
        return f"{quantity:.8f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_price(price: float) -> str:
        return f"{price:.8f}".rstrip("0").rstrip(".")

    @staticmethod
    def _offset_price(price: float, delta: float) -> float:
        return price * (1 + delta)

    @staticmethod
    def _extract_filled_price(order: dict) -> Optional[float]:
        fills = order.get("fills") or []
        if fills:
            return float(fills[-1]["price"])
        price = order.get("price")
        return float(price) if price else None

