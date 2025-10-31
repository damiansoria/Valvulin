"""Wrapper utilities to execute backtests with a unified return signature."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd

from analytics.backtest_visual import run_backtest as _run_visual_backtest


def run_backtest(
    *,
    data: pd.DataFrame,
    strategies: Sequence[str] | str,
    params: Dict[str, Dict[str, float]] | Dict[str, float] | None,
    capital_inicial: float = 1_000.0,
    riesgo_por_trade: float = 1.0,
    sl_ratio: float = 1.0,
    tp_ratio: float = 2.0,
    logica: str = "AND",
    symbol: str | None = None,
) -> Dict[str, Any]:
    """Execute a backtest and return serialisable results for the UI."""

    result = _run_visual_backtest(
        data,
        strategies,
        params,
        capital_inicial=capital_inicial,
        riesgo_por_trade=riesgo_por_trade,
        sl_ratio=sl_ratio,
        tp_ratio=tp_ratio,
        logica=logica,
        symbol=symbol,
    )

    equity_curve = result.equity_curve.rename("equity").reset_index()
    if equity_curve.columns.tolist() == ["index", "equity"]:
        equity_curve.rename(columns={"index": "timestamp"}, inplace=True)

    trades_df = result.trades.copy()

    return {
        "metrics": result.metrics,
        "equity": equity_curve,
        "trades": trades_df,
        "raw_result": result,
    }


__all__ = ["run_backtest"]
