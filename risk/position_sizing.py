"""Utilities for calculating position sizing based on risk parameters.

This module exposes helper functions to translate risk configuration
(percentage of capital at risk, stop distance, ATR, etc.) into
position sizes that can be used by execution layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PositionSizingResult:
    """Container with the numeric output of a position sizing call.

    Attributes
    ----------
    size:
        The raw position size that should be opened given the risk
        configuration. The value is expressed in units of the traded
        instrument (shares, contracts, lots, etc.).
    risk_amount:
        Absolute amount of capital that will be risked when opening the
        position. This corresponds to the monetary value for the
        specified risk percentage.
    stop_distance:
        The distance between the entry price and the stop loss in price
        terms. This can be derived from a stop price, ATR, or a fixed
        value depending on the parameters provided.
    """

    size: float
    risk_amount: float
    stop_distance: float


def calculate_position_size(
    account_balance: float,
    risk_percentage: float,
    entry_price: float,
    *,
    stop_loss_price: Optional[float] = None,
    stop_distance: Optional[float] = None,
    atr: Optional[float] = None,
    point_value: float = 1.0,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    rounding: Optional[float] = None,
) -> PositionSizingResult:
    """Calculate a position size based on a percentage of capital at risk.

    The function combines several common approaches for defining the
    distance to the stop loss:

    * Explicit stop loss price (``stop_loss_price``).
    * Fixed distance in price units (``stop_distance``).
    * ATR derived distance (``atr``).

    Parameters
    ----------
    account_balance:
        Total trading capital available.
    risk_percentage:
        Fraction of capital (expressed as a decimal) willing to be risked
        in the trade. For example, ``0.01`` corresponds to 1%.
    entry_price:
        Price at which the trade is expected to be executed.
    stop_loss_price:
        Absolute price of the stop loss. When provided, it takes
        precedence over other stop distance sources.
    stop_distance:
        Absolute distance between entry price and stop loss. Useful when
        the distance is defined in points/pips.
    atr:
        Average True Range. When no stop loss price or explicit stop
        distance are supplied, the stop distance is derived directly from
        the ATR value.
    point_value:
        Monetary value of a single price unit. For stocks it is typically
        1, while for futures or forex it depends on the contract specs.
    min_size / max_size:
        Optional hard bounds to clamp the resulting size.
    rounding:
        When provided, the resulting size is rounded to the nearest
        multiple of ``rounding`` (e.g. 0.01 for mini lots).

    Returns
    -------
    PositionSizingResult
        Dataclass with the calculated position size, risk amount and stop
        distance.
    """

    if account_balance <= 0:
        raise ValueError("Account balance must be positive.")
    if risk_percentage <= 0:
        raise ValueError("Risk percentage must be positive.")
    if entry_price <= 0:
        raise ValueError("Entry price must be positive.")
    if point_value <= 0:
        raise ValueError("Point value must be positive.")

    resolved_stop_distance: Optional[float] = None

    if stop_loss_price is not None:
        resolved_stop_distance = abs(entry_price - stop_loss_price)
    elif stop_distance is not None:
        resolved_stop_distance = stop_distance
    elif atr is not None:
        resolved_stop_distance = atr

    if resolved_stop_distance is None or resolved_stop_distance <= 0:
        raise ValueError(
            "A positive stop distance must be provided via stop_loss_price, "
            "stop_distance or atr parameters."
        )

    risk_amount = account_balance * risk_percentage
    per_unit_risk = resolved_stop_distance * point_value

    if per_unit_risk <= 0:
        raise ValueError("Per unit risk must be positive. Check the inputs.")

    raw_size = risk_amount / per_unit_risk

    if rounding:
        if rounding <= 0:
            raise ValueError("Rounding increment must be positive.")
        raw_size = round(raw_size / rounding) * rounding

    if min_size is not None:
        raw_size = max(raw_size, min_size)
    if max_size is not None:
        raw_size = min(raw_size, max_size)

    return PositionSizingResult(
        size=raw_size,
        risk_amount=risk_amount,
        stop_distance=resolved_stop_distance,
    )
