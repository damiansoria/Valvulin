"""Risk analytics helpers for reporting modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from risk.position_sizing import PositionSizingResult


@dataclass(frozen=True)
class RiskMetrics:
    """Bundle of risk related metrics for reporting."""

    value_at_risk: float
    capital_at_risk: float


def historical_var(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Compute a simple historical Value at Risk (VaR).

    The function expects a sequence of returns expressed as decimals
    (e.g. ``0.01`` = 1%). VaR is returned as a positive number
    representing the potential loss for the given confidence level.
    """

    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    if len(returns) == 0:
        raise ValueError("Returns sequence must not be empty.")

    sorted_returns = sorted(returns)
    index = max(int((1 - confidence) * len(sorted_returns)) - 1, 0)
    var = sorted_returns[index]
    return abs(min(var, 0.0))


def capital_at_risk(positions: Iterable[PositionSizingResult]) -> float:
    """Aggregate the risk amount for the provided positions."""

    return float(sum(max(result.risk_amount, 0.0) for result in positions))


def build_risk_metrics(
    returns: Sequence[float],
    positions: Iterable[PositionSizingResult],
    confidence: float = 0.95,
) -> RiskMetrics:
    """Construct a :class:`RiskMetrics` object combining VaR and CAR."""

    return RiskMetrics(
        value_at_risk=historical_var(returns, confidence=confidence),
        capital_at_risk=capital_at_risk(positions),
    )
