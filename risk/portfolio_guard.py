"""Portfolio level risk guardrails for automated strategies."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Position:
    """Simplified representation of an open position."""

    symbol: str
    notional: float
    correlation_group: Optional[str] = None


@dataclass
class PortfolioState:
    """Stores information about the portfolio across timeframes."""

    day_start_equity: float
    week_start_equity: float
    current_equity: float
    open_positions: List[Position] = field(default_factory=list)
    last_daily_reset: date = field(default_factory=date.today)
    last_weekly_reset: date = field(default_factory=date.today)


class PortfolioGuard:
    """Encapsulates portfolio-level risk controls.

    Parameters
    ----------
    total_exposure_limit:
        Maximum aggregate exposure allowed, expressed as a fraction of
        equity (e.g. 3.0 for 300% gross exposure).
    max_correlated_positions:
        Maximum number of simultaneously open positions within the same
        correlation group.
    daily_drawdown_limit / weekly_drawdown_limit:
        Maximum fractional loss from the start of the day/week before
        trading should halt (e.g. 0.03 for 3%).
    """

    def __init__(
        self,
        *,
        total_exposure_limit: float,
        max_correlated_positions: int,
        daily_drawdown_limit: float,
        weekly_drawdown_limit: float,
        initial_equity: float,
    ) -> None:
        if total_exposure_limit <= 0:
            raise ValueError("Total exposure limit must be positive.")
        if max_correlated_positions <= 0:
            raise ValueError("Max correlated positions must be positive.")
        if daily_drawdown_limit <= 0 or weekly_drawdown_limit <= 0:
            raise ValueError("Drawdown limits must be positive fractions.")
        if initial_equity <= 0:
            raise ValueError("Initial equity must be positive.")

        today = date.today()
        self.state = PortfolioState(
            day_start_equity=initial_equity,
            week_start_equity=initial_equity,
            current_equity=initial_equity,
            last_daily_reset=today,
            last_weekly_reset=today,
        )
        self.total_exposure_limit = total_exposure_limit
        self.max_correlated_positions = max_correlated_positions
        self.daily_drawdown_limit = daily_drawdown_limit
        self.weekly_drawdown_limit = weekly_drawdown_limit

    # ------------------------------------------------------------------
    # Portfolio updates
    def update_equity(self, current_equity: float, *, today: Optional[date] = None) -> None:
        """Update the equity snapshot, resetting day/week anchors if needed."""

        if current_equity <= 0:
            raise ValueError("Current equity must be positive.")

        today = today or date.today()
        self._maybe_reset_periods(today)
        self.state.current_equity = current_equity

    def _maybe_reset_periods(self, today: date) -> None:
        if today > self.state.last_daily_reset:
            self.state.day_start_equity = self.state.current_equity
            self.state.last_daily_reset = today
        if today.isocalendar()[1] > self.state.last_weekly_reset.isocalendar()[1] or today.year > self.state.last_weekly_reset.year:
            self.state.week_start_equity = self.state.current_equity
            self.state.last_weekly_reset = today

    def register_positions(self, positions: Iterable[Position]) -> None:
        """Replace the list of tracked positions."""

        self.state.open_positions = list(positions)

    # ------------------------------------------------------------------
    # Exposure checks
    def current_exposure(self) -> float:
        """Return the sum of absolute notionals for open positions."""

        return float(sum(abs(pos.notional) for pos in self.state.open_positions))

    def _exposure_limit(self) -> float:
        return self.state.current_equity * self.total_exposure_limit

    # ------------------------------------------------------------------
    # Correlation checks
    def correlated_position_counts(self) -> Dict[str, int]:
        """Return counts of positions grouped by correlation label."""

        groups = [pos.correlation_group or pos.symbol for pos in self.state.open_positions]
        return Counter(groups)

    # ------------------------------------------------------------------
    # Drawdown checks
    def daily_drawdown(self) -> float:
        return max(0.0, (self.state.day_start_equity - self.state.current_equity) / self.state.day_start_equity)

    def weekly_drawdown(self) -> float:
        return max(0.0, (self.state.week_start_equity - self.state.current_equity) / self.state.week_start_equity)

    # ------------------------------------------------------------------
    def check_limits(
        self,
        new_position: Optional[Position] = None,
    ) -> Dict[str, bool]:
        """Evaluate risk limits optionally including a prospective position.

        Returns a dictionary with the following boolean keys:

        ``total_exposure``
            True when the total exposure limit would be breached.
        ``correlated_positions``
            True when adding the new position would exceed the allowed
            number of correlated exposures.
        ``daily_drawdown`` / ``weekly_drawdown``
            True when the respective drawdown limit has been violated.
        """

        breaches = {
            "total_exposure": False,
            "correlated_positions": False,
            "daily_drawdown": False,
            "weekly_drawdown": False,
        }

        anticipated_exposure = self.current_exposure()
        if new_position is not None:
            anticipated_exposure += abs(new_position.notional)

        if anticipated_exposure > self._exposure_limit():
            breaches["total_exposure"] = True

        if new_position is not None:
            counts = self.correlated_position_counts()
            group = new_position.correlation_group or new_position.symbol
            counts[group] += 1
            if counts[group] > self.max_correlated_positions:
                breaches["correlated_positions"] = True

        if self.daily_drawdown() >= self.daily_drawdown_limit:
            breaches["daily_drawdown"] = True
        if self.weekly_drawdown() >= self.weekly_drawdown_limit:
            breaches["weekly_drawdown"] = True

        return breaches

    def can_open(self, position: Position) -> bool:
        """Convenience wrapper returning True when all limits pass."""

        breaches = self.check_limits(position)
        return not any(breaches.values())
