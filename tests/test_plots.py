"""Tests for analytics plotting helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.plots import plot_trade_signals
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
