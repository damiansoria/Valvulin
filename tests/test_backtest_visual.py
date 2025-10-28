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


def test_run_backtest_compounding_and_r_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    data = pd.DataFrame(
        {
            "open_time": pd.date_range("2023-02-01", periods=6, freq="h"),
            "close": [
                100.0,
                100.2,
                100.198,
                100.3,
                100.8,
                100.8,
            ],
        }
    )

    signals = pd.Series([1, 1, 0, 1, 1, 0], index=data.index, dtype=int)

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

    first_trade = trades.iloc[0]
    assert first_trade["pnl_usd"] == pytest.approx(0.99, rel=1e-8)
    assert first_trade["risk_usd"] == pytest.approx(10.0, rel=1e-9)
    assert first_trade["r_multiple"] == pytest.approx(
        first_trade["pnl_usd"] / first_trade["risk_usd"], rel=1e-9
    )
    assert first_trade["capital_final"] == pytest.approx(1_000.99, rel=1e-8)

    second_trade = trades.iloc[1]
    assert second_trade["capital_inicial"] == pytest.approx(
        first_trade["capital_final"], rel=1e-9
    )
    assert second_trade["risk_usd"] == pytest.approx(10.0099, rel=1e-6)
    assert second_trade["r_multiple"] == pytest.approx(
        second_trade["pnl_usd"] / second_trade["risk_usd"], rel=1e-9
    )

    final_equity = trades.iloc[-1]["capital_final"]
    assert result.equity_curve.iloc[-1] == pytest.approx(final_equity, rel=1e-9)
    assert result.equity_curve.iloc[0] == pytest.approx(1_000.0, rel=1e-9)
