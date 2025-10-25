"""Performance analytics helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from valvulin.core.logging import get_logger

logger = get_logger(__name__)


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    """Compute percentage returns from an equity curve."""

    returns = equity_curve.pct_change().fillna(0.0)
    logger.debug("analytics_compute_returns", extra={"periods": len(returns)})
    return returns


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the annualized Sharpe ratio."""

    excess = returns - (risk_free_rate / len(returns) if len(returns) else 0.0)
    std = excess.std()
    if std == 0:
        logger.warning("analytics_sharpe_division_by_zero")
        return 0.0
    ratio = np.sqrt(252) * excess.mean() / std
    logger.info("analytics_sharpe_ratio", extra={"ratio": ratio})
    return float(ratio)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return the maximum drawdown in percentage terms."""

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()
    logger.info("analytics_max_drawdown", extra={"max_drawdown": float(max_dd)})
    return float(max_dd)


__all__ = ["compute_returns", "sharpe_ratio", "max_drawdown"]
