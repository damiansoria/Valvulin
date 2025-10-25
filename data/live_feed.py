"""Real-time candle feed updated from Binance WebSocket streams."""
from __future__ import annotations

import asyncio
import datetime as dt
from collections import deque
from typing import Callable, Deque, Optional

import pandas as pd

from .binance_client import BinanceWebSocketClient, WebSocketConfig
from .loader import detect_lag, klines_to_dataframe, validate_continuity, validate_missing_values

CandleCallback = Callable[[pd.DataFrame], None]


class LiveCandleFeed:
    """Maintain a rolling window of candles updated via WebSocket."""

    def __init__(
        self,
        symbol: str,
        interval: str,
        max_candles: int = 500,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        on_update: Optional[CandleCallback] = None,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.max_candles = max_candles
        self._loop = loop or asyncio.get_event_loop()
        self._buffer: Deque[list] = deque(maxlen=max_candles)
        self._on_update = on_update
        self._ws_client = BinanceWebSocketClient(
            WebSocketConfig(symbol=symbol, interval=interval),
            message_handler=self._handle_message,
            loop=self._loop,
        )

    def start(self) -> None:
        self._ws_client.start()

    async def stop(self) -> None:
        await self._ws_client.stop()

    def _handle_message(self, message: dict) -> None:
        if "k" not in message:
            return
        kline = message["k"]
        candle = [
            kline["t"],
            kline["o"],
            kline["h"],
            kline["l"],
            kline["c"],
            kline["v"],
            kline["T"],
            kline["q"],
            kline["n"],
            kline["V"],
            kline["Q"],
            0,
        ]
        self._update_buffer(candle)
        if kline.get("x", False):
            self._finalize_and_emit()

    def _update_buffer(self, candle: list) -> None:
        open_time = candle[0]
        for idx, existing in enumerate(self._buffer):
            if existing[0] == open_time:
                self._buffer[idx] = candle
                break
        else:
            self._buffer.append(candle)

    def _finalize_and_emit(self) -> None:
        if self._on_update is None:
            return
        df = self.dataframe()
        self._on_update(df)

    def candles(self) -> list[list]:
        return list(self._buffer)

    def dataframe(self, validate: bool = True) -> pd.DataFrame:
        df = klines_to_dataframe(list(self._buffer))
        if validate and not df.empty:
            validate_continuity(df, self.interval)
            validate_missing_values(df)
        return df

    def lag(self, now: Optional[dt.datetime] = None) -> pd.Timedelta:
        return detect_lag(self.dataframe(validate=False), now=now)


__all__ = ["LiveCandleFeed"]
