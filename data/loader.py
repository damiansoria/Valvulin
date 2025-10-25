"""Utilities for downloading, validating and storing historical OHLCV data."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .binance_client import BinanceRESTClient


_INTERVAL_TO_MILLISECONDS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,
}


def interval_to_milliseconds(interval: str) -> int:
    try:
        return _INTERVAL_TO_MILLISECONDS[interval]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported interval: {interval}") from exc


def _to_milliseconds(timestamp: Optional[dt.datetime | int | float | str]) -> Optional[int]:
    if timestamp is None:
        return None
    if isinstance(timestamp, (int, float)):
        return int(timestamp)
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp, utc=True)
    if isinstance(timestamp, dt.datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
        return int(timestamp.timestamp() * 1000)
    raise TypeError(f"Unsupported timestamp type: {type(timestamp)!r}")


def download_klines(
    rest_client: BinanceRESTClient,
    symbol: str,
    interval: str,
    start_time: Optional[dt.datetime | int | float | str] = None,
    end_time: Optional[dt.datetime | int | float | str] = None,
    limit: int = 1000,
) -> list[list]:
    interval_ms = interval_to_milliseconds(interval)
    start_ms = _to_milliseconds(start_time)
    end_ms = _to_milliseconds(end_time)

    klines: list[list] = []
    fetch_from = start_ms
    while True:
        batch = rest_client.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=fetch_from,
            end_time=end_ms,
            limit=limit,
        )
        if not batch:
            break
        klines.extend(batch)
        last_open_time = batch[-1][0]
        fetch_from = last_open_time + interval_ms
        if end_ms is not None and fetch_from >= end_ms:
            break
        if len(batch) < limit:
            break
    return klines


def klines_to_dataframe(klines: Sequence[Sequence]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)
    if df.empty:
        return df
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    df.index.name = "timestamp"
    return df


def validate_continuity(df: pd.DataFrame, interval: str) -> None:
    if df.empty:
        return
    expected_delta = pd.Timedelta(milliseconds=interval_to_milliseconds(interval))
    deltas = df.index.to_series().diff().dropna()
    missing = deltas[deltas != expected_delta]
    if not missing.empty:
        raise ValueError(
            "Missing candles detected. Expected interval %s but got differences %s"
            % (expected_delta, missing.unique().tolist())
        )


def validate_missing_values(df: pd.DataFrame) -> None:
    if df.empty:
        return
    if df.isna().any().any():
        raise ValueError("OHLCV dataset contains missing values")


def detect_lag(df: pd.DataFrame, now: Optional[dt.datetime] = None) -> pd.Timedelta:
    if df.empty:
        return pd.Timedelta(0)
    now = now or dt.datetime.now(tz=dt.timezone.utc)
    last_close = df["close_time"].iloc[-1]
    if not isinstance(last_close, pd.Timestamp):
        last_close = pd.to_datetime(last_close, utc=True)
    return now - last_close


def store_dataframe_to_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def store_dataframe_to_sqlite(
    df: pd.DataFrame,
    path: str | Path,
    table: str = "ohlcv",
    if_exists: str = "append",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=True, index_label="timestamp")
    return path


def download_and_store(
    symbol: str,
    interval: str,
    start_time: Optional[dt.datetime | int | float | str] = None,
    end_time: Optional[dt.datetime | int | float | str] = None,
    storage: str | Path | None = None,
    storage_format: str = "csv",
    rest_client: Optional[BinanceRESTClient] = None,
) -> pd.DataFrame:
    owns_client = rest_client is None
    rest_client = rest_client or BinanceRESTClient()
    try:
        klines = download_klines(rest_client, symbol, interval, start_time, end_time)
        df = klines_to_dataframe(klines)
        validate_continuity(df, interval)
        validate_missing_values(df)

        if storage:
            if storage_format == "csv":
                store_dataframe_to_csv(df, storage)
            elif storage_format == "sqlite":
                store_dataframe_to_sqlite(df, storage)
            else:  # pragma: no cover - defensive branch
                raise ValueError(f"Unsupported storage_format: {storage_format}")
        return df
    finally:
        if owns_client:
            rest_client.close()


__all__ = [
    "download_klines",
    "klines_to_dataframe",
    "validate_continuity",
    "validate_missing_values",
    "detect_lag",
    "store_dataframe_to_csv",
    "store_dataframe_to_sqlite",
    "download_and_store",
]
