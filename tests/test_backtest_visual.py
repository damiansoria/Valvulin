from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.backtest_visual import run_backtest


def test_run_backtest_updates_equity_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    data = pd.DataFrame(
        {
            "open_time": pd.date_range("2023-01-01", periods=5, freq="D"),
            "close": [100.0, 105.0, 102.0, 107.0, 103.0],
        }
    )

    signals = pd.Series([1, 1, -1, -1, 0], index=data.index, dtype=int)

    def fake_signal(df: pd.DataFrame, strategy: str, params: dict[str, float]) -> pd.Series:
        return signals.copy()

    monkeypatch.setattr(
        "analytics.backtest_visual._strategy_signal", fake_signal
    )

    result = run_backtest(
        data,
        strategies=["Fake"],
        params={},
        capital_inicial=1_000.0,
        riesgo_por_trade=1.0,
        stop_loss_pct=2.0,
        logica="OR",
    )

    trades = result.trades
    assert len(trades) == 2
    assert trades.loc[0, "pnl_usd"] == pytest.approx(10.0, rel=1e-9)
    assert trades.loc[1, "pnl_usd"] == pytest.approx(-4.95098039, rel=1e-6)
    assert trades.loc[0, "position_size"] == pytest.approx(5.0, rel=1e-9)
    assert trades.loc[1, "position_size"] == pytest.approx(4.95098039, rel=1e-6)
    assert trades.loc[1, "capital_inicial"] == pytest.approx(
        trades.loc[0, "capital_final"], rel=1e-9
    )

    final_equity = trades.loc[1, "capital_final"]
    assert result.equity_curve.iloc[-1] == pytest.approx(final_equity, rel=1e-9)
    assert np.isclose(result.drawdown.iloc[-1], -0.004905, atol=1e-4)
    assert result.drawdown.min() < 0

    metrics = result.metrics
    assert metrics["Total Trades"] == 2
    assert metrics["Winning Trades"] == 1
    assert metrics["Losing Trades"] == 1
    assert metrics["Equity Final $"] == pytest.approx(round(final_equity, 2), rel=1e-9)
    assert metrics["Total Return $"] == pytest.approx(round(final_equity - 1_000.0, 2), rel=1e-9)
    assert metrics["Expectancy $"] == pytest.approx(2.52, rel=1e-2)
    assert metrics["Max Drawdown %"] == pytest.approx(0.49, rel=1e-2)
