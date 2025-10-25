"""Unit tests for strategy implementations using synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategies import (  # noqa: E402  # isort:skip
    BreakoutVolumeStrategy,
    CandlestickPatternStrategy,
    EMAMACDStrategy,
    InsideBarStrategy,
    PullbackEMAStrategy,
    RSIDivergenceStrategy,
)


def test_breakout_volume_signal_triggered() -> None:
    data = [
        {"high": 100, "close": 95, "volume": 100},
        {"high": 102, "close": 101, "volume": 120},
        {"high": 103, "close": 102, "volume": 130},
        {"high": 104, "close": 106, "volume": 260},
    ]
    strategy = BreakoutVolumeStrategy(breakout_window=3, volume_multiplier=2.0)
    assert strategy.generate_signal(data) == "buy"


def test_breakout_volume_no_volume_confirmation() -> None:
    data = [
        {"high": 100, "close": 99, "volume": 200},
        {"high": 101, "close": 100, "volume": 200},
        {"high": 102, "close": 103, "volume": 150},
    ]
    strategy = BreakoutVolumeStrategy(breakout_window=2, volume_multiplier=1.5)
    assert strategy.generate_signal(data) is None


def test_pullback_ema_strategy_detects_resumption() -> None:
    data = [
        {"close": value}
        for value in [100, 101, 102, 103, 104, 105, 104, 108]
    ]
    strategy = PullbackEMAStrategy(short_window=3, long_window=5)
    assert strategy.generate_signal(data) == "buy"


def test_candlestick_bullish_engulfing() -> None:
    data = [
        {"open": 105, "close": 100},
        {"open": 99, "close": 110},
    ]
    strategy = CandlestickPatternStrategy(pattern="bullish_engulfing")
    assert strategy.generate_signal(data) == "buy"


@pytest.mark.parametrize(
    "pattern, expected",
    [("bullish_engulfing", "buy"), ("bearish_engulfing", "sell")],
)
def test_candlestick_configurable_patterns(pattern: str, expected: str) -> None:
    first = {"open": 10, "close": 8}
    second = {"open": 7, "close": 11}
    if pattern == "bearish_engulfing":
        first, second = {"open": 8, "close": 10}, {"open": 11, "close": 7}
    strategy = CandlestickPatternStrategy(pattern=pattern)
    assert strategy.generate_signal([first, second]) == expected


def test_ema_macd_generates_buy_signal() -> None:
    data = [
        {"close": price}
        for price in [100, 99, 98, 97, 96, 97, 98, 99]
    ]
    strategy = EMAMACDStrategy(ema_short=3, ema_long=5, macd_signal=3)
    assert strategy.generate_signal(data) == "buy"


def test_inside_bar_identification() -> None:
    data = [
        {"high": 110, "low": 100},
        {"high": 112, "low": 98},
        {"high": 109, "low": 103},
    ]
    strategy = InsideBarStrategy(max_ratio=0.6)
    assert strategy.generate_signal(data) == "watch"


def test_rsi_divergence_bullish_and_bearish() -> None:
    bullish_data = [
        {"close": 100, "rsi": 30},
        {"close": 98, "rsi": 35},
    ]
    bearish_data = [
        {"close": 100, "rsi": 60},
        {"close": 102, "rsi": 55},
    ]
    bullish = RSIDivergenceStrategy(divergence_type="both")
    bearish = RSIDivergenceStrategy(divergence_type="bearish")
    assert bullish.generate_signal(bullish_data) == "buy"
    assert bearish.generate_signal(bearish_data) == "sell"


def test_strategy_state_updates() -> None:
    strategy = BreakoutVolumeStrategy()
    strategy.update_state({"last_trade": "buy"})
    assert strategy.state["last_trade"] == "buy"
