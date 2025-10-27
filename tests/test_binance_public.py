"""Pruebas básicas para el feed público de Binance."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from valvulin.data.binance_public import BinancePublicDataFeed  # noqa: E402  # isort:skip


def test_fetch_klines_basic(monkeypatch):
    feed = BinancePublicDataFeed(rate_sleep=0)
    sample_response = [
        [1704067200000, "1", "2", "0.5", "1.5", "100"],
        [1704070800000, "1.5", "2.5", "1.0", "2.0", "150"],
    ]

    monkeypatch.setattr(feed, "_request", lambda url, params: sample_response)
    data = feed.fetch_klines(
        "BTCUSDT",
        "1h",
        start_time=1704067200000,
        end_time=1704070800000,
    )
    assert isinstance(data, pd.DataFrame)
    for column in ["open_time", "open", "high", "low", "close", "volume"]:
        assert column in data.columns


def test_fetch_klines_progress_callback(monkeypatch):
    feed = BinancePublicDataFeed(rate_sleep=0, max_limit=500)
    sample_response = [
        [1704067200000, "1", "2", "0.5", "1.5", "100"],
        [1704070800000, "1.5", "2.5", "1.0", "2.0", "150"],
    ]

    responses = iter([sample_response, []])
    monkeypatch.setattr(feed, "_request", lambda url, params: next(responses))

    calls = []

    def progress(chunk, total_rows):
        calls.append((chunk.count, total_rows))

    data = feed.fetch_klines(
        "ETHUSDT",
        "1h",
        start_time=1704067200000,
        end_time=1704070800000,
        progress_callback=progress,
    )

    assert isinstance(data, pd.DataFrame)
    assert calls
    assert calls[0][0] == 2
    assert calls[0][1] == 2
