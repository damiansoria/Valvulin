"""Binance API clients with reconnection handling for REST and WebSocket access."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import requests
import websockets
from websockets import WebSocketClientProtocol

logger = logging.getLogger(__name__)


class BinanceRESTClient:
    """Thin REST client that retries on transient failures."""

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        timeout: float = 10.0,
        max_retries: int = 5,
        retry_delay: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.request(method, url, timeout=self.timeout, **kwargs)
                if response.status_code == 429:
                    # API rate limit hit, apply backoff
                    retry_after = float(response.headers.get("Retry-After", self.retry_delay))
                    logger.warning("Rate limited by Binance API. Sleeping for %s seconds", retry_after)
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.max_retries:
                    logger.exception("Binance REST request failed after %s attempts", attempt)
                    raise
                delay = self.retry_delay * attempt
                logger.warning("REST request failed (%s). Retrying in %.2fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError("Unreachable retry loop")

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        params: Dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        response = self._request("GET", "/api/v3/klines", params=params)
        return response.json()

    def close(self) -> None:
        self._session.close()


@dataclass
class WebSocketConfig:
    symbol: str
    interval: str = "1m"
    stream_type: str = "kline"
    base_url: str = "wss://stream.binance.com:9443/ws"
    ping_interval: float = 20.0
    ping_timeout: float = 20.0
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 30.0

    def stream_name(self) -> str:
        return f"{self.symbol.lower()}@{self.stream_type}_{self.interval}"

    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.stream_name()}"


class BinanceWebSocketClient:
    """Manage a Binance WebSocket connection with automatic reconnection."""

    def __init__(
        self,
        config: WebSocketConfig,
        message_handler: Optional[Callable[[dict[str, Any]], Awaitable[None] | None]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.config = config
        self._loop = loop or asyncio.get_event_loop()
        self._handler = message_handler
        self._running = False
        self._ws: Optional[WebSocketClientProtocol] = None
        self._task: Optional[asyncio.Task[None]] = None

    async def _handle_messages(self, ws: WebSocketClientProtocol) -> None:
        async for raw_message in ws:
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.debug("Received non-JSON payload: %s", raw_message)
                continue
            if self._handler is None:
                continue
            result = self._handler(payload)
            if asyncio.iscoroutine(result):
                await result

    async def _connect_loop(self) -> None:
        delay = self.config.reconnect_delay
        while self._running:
            try:
                logger.info("Connecting to Binance WebSocket %s", self.config.url())
                async with websockets.connect(
                    self.config.url(),
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                ) as ws:
                    self._ws = ws
                    delay = self.config.reconnect_delay
                    await self._handle_messages(ws)
            except asyncio.CancelledError:
                logger.info("Binance WebSocket task cancelled")
                raise
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("WebSocket connection error: %s", exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.max_reconnect_delay)
            finally:
                self._ws = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = self._loop.create_task(self._connect_loop())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._ws:
            await self._ws.close()
        self._task = None
        self._ws = None


__all__ = [
    "BinanceRESTClient",
    "BinanceWebSocketClient",
    "WebSocketConfig",
]
