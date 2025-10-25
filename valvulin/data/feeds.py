"""Market data access layer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from valvulin.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketDataRequest:
    """Parameters describing a historical data request."""

    symbol: str
    interval: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None


class CSVDataFeed:
    """Simple CSV-based data feed for offline analytics and backtesting."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def load(self, request: MarketDataRequest) -> pd.DataFrame:
        file_path = self.data_dir / f"{request.symbol}_{request.interval}.csv"
        logger.info("loading_csv_data", extra={"file_path": str(file_path), "symbol": request.symbol})
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        frame = pd.read_csv(file_path, parse_dates=["open_time"], infer_datetime_format=True)
        if request.start:
            frame = frame[frame["open_time"] >= request.start]
        if request.end:
            frame = frame[frame["open_time"] <= request.end]
        return frame


def resample_to_interval(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample OHLCV data to a new interval using pandas."""

    logger.debug("resample_data", extra={"interval": interval, "rows": len(data)})
    ohlcv_columns = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    resampled = data.resample(interval, on="open_time").agg(ohlcv_columns).dropna()
    resampled.reset_index(inplace=True)
    return resampled


__all__ = ["MarketDataRequest", "CSVDataFeed", "resample_to_interval"]
