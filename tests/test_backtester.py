import pandas as pd
import pytest

from execution.backtester import Backtester, BacktestSettings
from execution.types import TradeSignal


def _sample_ohlcv() -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=8, freq="h")
    data = {
        "open": [100, 101, 102, 104, 105, 106, 107, 108],
        "high": [101, 104, 105, 106, 107, 108, 109, 110],
        "low": [99, 100, 101, 103, 104, 105, 106, 107],
        "close": [100, 103, 104, 104, 106, 107, 108, 109],
    }
    return pd.DataFrame(data, index=index)


def test_backtester_updates_equity_with_risk_management():
    ohlcv = _sample_ohlcv()
    backtester = Backtester(ohlcv)

    signals = [
        TradeSignal(
            symbol="BTCUSDT",
            timestamp=ohlcv.index[0].to_pydatetime(),
            side="BUY",
            quantity=0.0,
            stop_loss=98.0,
            take_profit=104.0,
        ),
        TradeSignal(
            symbol="BTCUSDT",
            timestamp=ohlcv.index[3].to_pydatetime(),
            side="SELL",
            quantity=0.0,
            stop_loss=106.0,
            take_profit=None,
        ),
    ]

    settings = BacktestSettings(
        risk_per_trade_pct=1.0,
        rr_ratio=2.0,
        sl_ratio=1.0,
        commission_pct=0.0,
    )
    result = backtester.run(signals, settings=settings, initial_equity=1_000.0)

    assert result.total_trades == 2
    assert result.win_rate == pytest.approx(result.winning_trades / result.total_trades)

    # Cada operación debe respetar el 1% de riesgo respecto al capital de entrada.
    for trade in result.trades:
        max_risk_allowed = trade.entry_equity * settings.risk_per_trade_pct / 100
        assert trade.risk_amount == pytest.approx(max_risk_allowed, rel=1e-2, abs=1e-6)

    assert result.equity_at_risk == pytest.approx(10.0, rel=1e-2)
    assert isinstance(result.strategy_r_distribution, dict)
    assert any(not series.empty for series in result.strategy_r_distribution.values())
    assert result.sharpe_ratio <= 0  # estrategia pierde en este escenario sintético
