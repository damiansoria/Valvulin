"""Data feed management for Valvulin."""
from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import DataFeedConfig


class DataFeedError(RuntimeError):
    """Raised when a data feed fails."""


class BaseDataFeed:
    """Base class for all data feed connectors."""

    def __init__(self, name: str, symbols: Sequence[str], **kwargs: Any) -> None:
        self.name = name
        self.symbols = list(symbols)
        self.kwargs = kwargs
        self.logger = logging.getLogger(f"valvulin.data_feed.{name}")

    async def connect(self) -> None:
        """Connect to the remote exchange or data source."""

    async def disconnect(self) -> None:
        """Disconnect and cleanup resources."""

    async def stream(self, queue: "asyncio.Queue[Dict[str, Any]]") -> None:
        """Stream incoming events into the shared queue."""
        raise NotImplementedError


class MockDataFeed(BaseDataFeed):
    """A mock feed that generates fake OHLC candles."""

    def __init__(self, name: str, symbols: Sequence[str], interval: float = 1.0, **kwargs: Any) -> None:
        super().__init__(name, symbols, **kwargs)
        self.interval = interval
        self._running = False

    async def connect(self) -> None:  # pragma: no cover - demonstration only
        self._running = True
        self.logger.info("Mock feed '%s' connected", self.name)

    async def disconnect(self) -> None:  # pragma: no cover - demonstration only
        self._running = False
        self.logger.info("Mock feed '%s' disconnected", self.name)

    async def stream(self, queue: "asyncio.Queue[Dict[str, Any]]") -> None:
        await self.connect()
        try:
            candle_id = 0
            while self._running:
                for symbol in self.symbols:
                    candle_id += 1
                    payload = {
                        "symbol": symbol,
                        "timeframe": self.kwargs.get("timeframe", "1m"),
                        "open": 100 + candle_id,
                        "close": 100 + candle_id + 1,
                        "high": 100 + candle_id + 2,
                        "low": 100 + candle_id - 2,
                        "volume": candle_id * 10,
                        "source": self.name,
                    }
                    await queue.put(payload)
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:  # pragma: no cover
            pass
        finally:
            await self.disconnect()


def _import_handler(path: str) -> type:
    module_name, _, attr = path.partition(":")
    if not attr:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


@dataclass
class DataFeedInstance:
    config: DataFeedConfig
    handler: BaseDataFeed
    task: Optional["asyncio.Task[None]"] = None


class DataFeedManager:
    def __init__(self, configs: Iterable[DataFeedConfig], queue: "asyncio.Queue[Dict[str, Any]]", loop: Optional[asyncio.AbstractEventLoop] = None, logger: Optional[logging.Logger] = None) -> None:
        self.loop = loop or asyncio.get_event_loop()
        self.queue = queue
        self.logger = logger or logging.getLogger("valvulin.data_feed")
        self.feeds: List[DataFeedInstance] = []
        for config in configs:
            if not config.handler:
                self.logger.warning(
                    "Skipping data feed '%s' without handler (type=%s)",
                    config.name,
                    config.type,
                )
                continue
            handler_cls = _import_handler(config.handler)
            handler = handler_cls(config.name, config.symbols, **config.parameters)
            self.feeds.append(DataFeedInstance(config=config, handler=handler))

    async def start(self) -> None:
        for instance in self.feeds:
            if instance.task and not instance.task.done():
                continue
            instance.task = self.loop.create_task(self._run_feed(instance))
            self.logger.info("Data feed '%s' started", instance.config.name)

    async def stop(self) -> None:
        for instance in self.feeds:
            if instance.task:
                instance.task.cancel()
        tasks = [instance.task for instance in self.feeds if instance.task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for instance in self.feeds:
            instance.task = None

    async def _run_feed(self, instance: DataFeedInstance) -> None:
        try:
            await instance.handler.stream(self.queue)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - logging side effect
            self.logger.exception("Data feed '%s' crashed", instance.config.name)

