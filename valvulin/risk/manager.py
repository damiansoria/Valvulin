"""Risk management primitives."""
from __future__ import annotations

from dataclasses import dataclass

from valvulin.core.config import AppConfig
from valvulin.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PositionState:
    """Simple container describing an open position."""

    symbol: str
    quantity: float
    entry_price: float
    pnl_pct: float


class RiskManager:
    """Encapsulates risk checks before orders are routed."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def check_position_size(self, position: PositionState, account_equity: float) -> bool:
        """Ensure the position size does not exceed configured limits."""

        value = position.quantity * position.entry_price
        limit = account_equity * (self.config.risk.max_position_size_pct / 100)
        allowed = value <= limit
        logger.info(
            "risk_check_position_size",
            extra={
                "symbol": position.symbol,
                "value": value,
                "limit": limit,
                "allowed": allowed,
            },
        )
        return allowed

    def check_drawdown(self, daily_loss_pct: float) -> bool:
        """Verify daily loss does not exceed threshold."""

        allowed = daily_loss_pct <= self.config.risk.max_daily_loss_pct
        logger.warning(
            "risk_check_drawdown" if not allowed else "risk_drawdown_within_limits",
            extra={"daily_loss_pct": daily_loss_pct, "threshold": self.config.risk.max_daily_loss_pct},
        )
        return allowed


__all__ = ["RiskManager", "PositionState"]
