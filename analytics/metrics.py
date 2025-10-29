"""Risk analytics helpers for reporting modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

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


def trade_distribution_metrics(r_values: Sequence[float]) -> dict[str, float]:
    """Compute expectancy-focused statistics from a sequence of R multiples."""

    r_array = np.array(list(r_values), dtype=float)
    if r_array.size == 0:
        return {
            "Average Win (R)": 0.0,
            "Average Loss (R)": 0.0,
            "RR Effective": 0.0,
            "Breakeven Winrate %": 0.0,
            "Expectancy (R)": 0.0,
            "Winrate": 0.0,
        }

    wins = r_array[r_array > 0]
    losses = r_array[r_array < 0]

    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0

    winrate = float(wins.size) / float(r_array.size)
    rr_effective = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")
    breakeven_winrate = (
        abs(avg_loss) / (abs(avg_loss) + avg_win) if (avg_win + abs(avg_loss)) > 0 else 0.0
    )
    expectancy = winrate * avg_win + (1 - winrate) * avg_loss

    return {
        "Average Win (R)": avg_win,
        "Average Loss (R)": avg_loss,
        "RR Effective": rr_effective if np.isfinite(rr_effective) else float("inf"),
        "Breakeven Winrate %": breakeven_winrate * 100,
        "Expectancy (R)": expectancy,
        "Winrate": winrate,
    }


__all__ = [
    "RiskMetrics",
    "historical_var",
    "capital_at_risk",
    "build_risk_metrics",
    "trade_distribution_metrics",
]
