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
        initial_equity=1_000.0,
        risk_per_trade_pct=1.0,
        commission_pct=0.0,
        maker_commission_pct=0.0,
        taker_commission_pct=0.0,
        use_trailing_stop=False,
        min_volume_ratio=0.0,
        volatility_atr_threshold=None,
        export_directory=None,
    )
    result = backtester.run(signals, settings=settings)

    assert pytest.approx(result.final_equity, rel=1e-6) == 1009.8
    assert result.total_trades == 2
    assert result.winning_trades == 1
    assert result.losing_trades == 1
    assert pytest.approx(result.win_rate, rel=1e-6) == 0.5

    equity_series = result.equity_curve
    assert pytest.approx(float(equity_series.iloc[0]), rel=1e-9) == 1000.0
    assert pytest.approx(float(equity_series.iloc[-1]), rel=1e-9) == 1009.8

    drawdown_series = result.drawdown_curve
    expected_drawdown = 10.2 / 1020.0
    assert float(drawdown_series.iloc[-1]) == pytest.approx(expected_drawdown, rel=1e-6)

    trades = result.trades
    assert len(trades) == 2
    first_trade, second_trade = trades
    assert pytest.approx(first_trade.quantity, rel=1e-6) == 5.0
    assert pytest.approx(first_trade.pnl, rel=1e-6) == 20.0
    assert pytest.approx(first_trade.risk_amount, rel=1e-6) == 10.0
    assert pytest.approx(first_trade.pnl_pct, rel=1e-6) == 0.02

    assert pytest.approx(second_trade.quantity, rel=1e-6) == 5.1
    assert pytest.approx(second_trade.pnl, rel=1e-6) == -10.2
    assert pytest.approx(second_trade.risk_amount, rel=1e-6) == 10.2
    assert pytest.approx(result.cumulative_return, rel=1e-6) == 9.8
    assert pytest.approx(result.cumulative_return_pct, rel=1e-6) == 0.98
