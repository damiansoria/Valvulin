"""Helpers to derive stop loss, take profit and trailing stop levels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class TrailingStop:
    """Configuration and resolved values for a trailing stop."""

    type: str
    distance: float
    activation_price: Optional[float] = None


@dataclass(frozen=True)
class OrderLevels:
    """Aggregated structure with stop, take profit and trailing levels."""

    entry_price: float
    stop_loss: float
    take_profit: Dict[str, float]
    risk_reward: Dict[str, float]
    trailing_stop: Optional[TrailingStop]


def _resolve_trailing_stop(
    entry_price: float,
    direction: str,
    stop_distance: float,
    trailing_config: Optional[Dict[str, float]],
    atr: Optional[float],
) -> Optional[TrailingStop]:
    if not trailing_config:
        return None

    trailing_type = trailing_config.get("type", "fixed").lower()
    activation_rr = trailing_config.get("activation_rr")

    if trailing_type == "fixed":
        distance = trailing_config.get("value", stop_distance)
    elif trailing_type == "percentage":
        percent = trailing_config.get("value")
        if percent is None:
            raise ValueError("Percentage trailing stop requires a 'value'.")
        distance = entry_price * percent
    elif trailing_type == "atr":
        if atr is None:
            raise ValueError("ATR trailing stop requires the 'atr' argument.")
        multiplier = trailing_config.get("value", 1.0)
        distance = atr * multiplier
    else:
        raise ValueError(f"Unsupported trailing stop type: {trailing_type}")

    if distance <= 0:
        raise ValueError("Trailing stop distance must be positive.")

    activation_price = None
    if activation_rr is not None:
        sign = 1 if direction == "long" else -1
        activation_price = entry_price + sign * activation_rr * stop_distance

    return TrailingStop(type=trailing_type, distance=distance, activation_price=activation_price)


def generate_order_levels(
    entry_price: float,
    stop_distance: float,
    *,
    direction: str = "long",
    risk_reward_targets: Iterable[float] = (2.0, 3.0),
    trailing_config: Optional[Dict[str, float]] = None,
    atr: Optional[float] = None,
) -> OrderLevels:
    """Generate stop loss, take profit and trailing stop levels.

    Parameters
    ----------
    entry_price:
        Planned execution price for the order.
    stop_distance:
        Distance between entry and stop in price units (points/pips).
    direction:
        Either ``"long"`` or ``"short"``. Determines the sign of the
        take profit calculations.
    risk_reward_targets:
        Iterable of risk/reward multiples for which to compute take profit
        prices. Typical values include ``(2.0, 3.0)`` representing 2R and
        3R targets.
    trailing_config:
        Optional dictionary describing trailing stop behaviour. Supported
        keys:

        ``type``: ``"fixed"`` (default), ``"percentage"`` or ``"atr"``.
        ``value``: magnitude associated to the type.
        ``activation_rr``: risk multiple at which the trailing stop starts
            following price.
    atr:
        Latest Average True Range value. Required when ``type='atr'`` for
        the trailing stop.
    """

    if entry_price <= 0:
        raise ValueError("Entry price must be positive.")
    if stop_distance <= 0:
        raise ValueError("Stop distance must be positive.")

    direction = direction.lower()
    if direction not in {"long", "short"}:
        raise ValueError("Direction must be either 'long' or 'short'.")

    sign = 1 if direction == "long" else -1

    stop_loss = entry_price - sign * stop_distance

    take_profit: Dict[str, float] = {}
    risk_reward: Dict[str, float] = {}
    for rr in risk_reward_targets:
        if rr <= 0:
            raise ValueError("Risk/reward targets must be positive.")
        price = entry_price + sign * stop_distance * rr
        key = f"{rr:.1f}:1"
        take_profit[key] = price
        risk_reward[key] = rr

    trailing_stop = _resolve_trailing_stop(
        entry_price=entry_price,
        direction=direction,
        stop_distance=stop_distance,
        trailing_config=trailing_config,
        atr=atr,
    )

    return OrderLevels(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=risk_reward,
        trailing_stop=trailing_stop,
    )
