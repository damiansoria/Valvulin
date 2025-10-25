"""Tests for analytics plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import analytics.signals_plot as signals_plot
from analytics.plots import plot_trade_signals
from analytics.signals_plot import plot_backtest_signals
from execution.backtester import TradeRecord
from execution.types import OrderStatus, TradeSignal


def _ema(span: int):
    def compute(df: pd.DataFrame) -> pd.Series:
        return df["close"].ewm(span=span, adjust=False).mean()

    return compute


def _macd(short_span: int, long_span: int, signal_span: int):
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        short = df["close"].ewm(span=short_span, adjust=False).mean()
        long = df["close"].ewm(span=long_span, adjust=False).mean()
        macd_line = short - long
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
        return pd.DataFrame({"macd": macd_line, "signal": signal_line})

    return compute


def test_plot_trade_signals_adds_annotations_and_indicators() -> None:
    index = pd.date_range("2023-01-01", periods=6, freq="1h")
    ohlc = pd.DataFrame(
        {
            "open": [100, 102, 101, 104, 103, 106],
            "high": [103, 104, 103, 107, 105, 108],
            "low": [99, 100, 99, 102, 101, 104],
            "close": [102, 101, 104, 103, 105, 107],
        },
        index=index,
    )

    buy_signal = TradeSignal(
        symbol="BTCUSDT",
        timestamp=index[1].to_pydatetime(),
        side="BUY",
        quantity=1.0,
        stop_loss=99.0,
    )
    sell_signal = TradeSignal(
        symbol="BTCUSDT",
        timestamp=index[2].to_pydatetime(),
        side="SELL",
        quantity=2.0,
        stop_loss=108.0,
    )

    trades = [
        TradeRecord(
            signal=buy_signal,
            entry_time=index[1].to_pydatetime(),
            exit_time=index[3].to_pydatetime(),
            entry_price=101.5,
            exit_price=105.0,
            quantity=1.0,
            pnl=3.5,
            r_multiple=1.2,
            status=OrderStatus.FILLED,
        ),
        TradeRecord(
            signal=sell_signal,
            entry_time=index[2].to_pydatetime(),
            exit_time=index[4].to_pydatetime(),
            entry_price=103.5,
            exit_price=106.0,
            quantity=2.0,
            pnl=-5.0,
            r_multiple=-0.8,
            status=OrderStatus.FILLED,
        ),
    ]

    figure = plot_trade_signals(
        ohlc,
        trades,
        indicator_builders={
            "EMA(3)": _ema(3),
            "MACD": _macd(short_span=2, long_span=4, signal_span=3),
        },
    )

    ax = figure.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    legend_entries = dict(zip(labels, handles))

    assert "Profitable Buy entry" in legend_entries
    assert "Losing Sell entry" in legend_entries
    assert "Trade exit" in legend_entries
    assert "Stop loss" in legend_entries
    assert "EMA(3)" in legend_entries
    assert "MACD (macd)" in legend_entries
    assert "MACD (signal)" in legend_entries

    pnl_annotations = [text for text in ax.texts if text.get_text().startswith("PnL")]
    assert len(pnl_annotations) == len(trades)

    plt.close(figure)


def test_plot_backtest_signals_generates_html(tmp_path: Path, monkeypatch) -> None:
    index = pd.date_range("2023-01-01", periods=5, freq="1h")
    ohlc = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
        },
        index=index,
    )

    trades = [
        {
            "entry_time": index[1],
            "exit_time": index[2],
            "entry_price": 101.5,
            "exit_price": 103.0,
            "side": "BUY",
            "pnl": 1.5,
        },
        {
            "entry_time": index[3],
            "exit_time": index[4],
            "entry_price": 103.5,
            "exit_price": 102.0,
            "side": "SELL",
            "pnl": -1.5,
        },
    ]

    monkeypatch.setattr(signals_plot, "PLOTS_DIR", tmp_path)
    output = plot_backtest_signals(ohlc, trades, title="Test Backtest")

    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "Test Backtest" in content


def test_plot_backtest_signals_renders_buy_and_sell_markers(
    tmp_path: Path, monkeypatch
) -> None:
    index = pd.date_range("2023-01-01", periods=6, freq="1h")
    ohlc = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104, 105],
            "High": [101, 102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103, 104],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
        },
        index=index,
    )

    trades = []
    for offset, side in enumerate(["BUY", "SELL", "BUY", "SELL"], start=1):
        trades.append(
            {
                "entry_time": index[offset],
                "exit_time": index[offset],
                "entry_price": 100 + offset,
                "exit_price": 100 + offset + (1 if side == "BUY" else -1),
                "side": side,
                "pnl": 1.0 if side == "BUY" else -1.0,
            }
        )

    monkeypatch.setattr(signals_plot, "PLOTS_DIR", tmp_path)
    output = plot_backtest_signals(ohlc, trades, title="Markers Test")
    html = output.read_text(encoding="utf-8")
    assert html.count("triangle-up") >= 2
    assert html.count("triangle-down") >= 2
