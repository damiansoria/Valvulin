"""Helpers to derive stop loss and take profit levels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class OrderLevels:
    """Aggregated structure with stop and take profit levels."""

    entry_price: float
    stop_loss: float
    take_profit: Dict[str, float]
    risk_reward: Dict[str, float]


def generate_order_levels(
    entry_price: float,
    stop_distance: float,
    *,
    direction: str = "long",
    risk_reward_targets: Iterable[float] = (2.0, 3.0),
) -> OrderLevels:
    """Generate stop loss and take profit levels.

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

    return OrderLevels(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=risk_reward,
    )
