"""Strategy abstractions for Valvulin."""
from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import StrategyConfig


class StrategyError(RuntimeError):
    """Raised when a strategy fails."""


class BaseStrategy:
    """Base class for trading strategies."""

    def __init__(self, config: StrategyConfig, risk_manager: "RiskManager", execution_engine: "ExecutionEngine") -> None:
        self.config = config
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.logger = logging.getLogger(f"valvulin.strategy.{config.name}")

    async def on_market_data(self, event: Dict[str, Any]) -> None:
        """Process a market data event."""

    async def on_scheduled_event(self, event: Dict[str, Any]) -> None:
        """Called by the scheduler for timeframe based evaluation."""


@dataclass
class StrategyInstance:
    config: StrategyConfig
    handler: BaseStrategy

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.config.symbol, self.config.timeframe, self.config.name)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled


def _import_handler(path: str) -> type:
    module_name, _, attr = path.partition(":")
    if not attr:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class StrategyManager:
    def __init__(self, configs: Iterable[StrategyConfig], risk_manager: "RiskManager", execution_engine: "ExecutionEngine", loop: Optional[asyncio.AbstractEventLoop] = None, logger: Optional[logging.Logger] = None) -> None:
        self.loop = loop or asyncio.get_event_loop()
        self.logger = logger or logging.getLogger("valvulin.strategy")
        self.instances: List[StrategyInstance] = []
        for config in configs:
            handler_cls = _import_handler(config.handler)
            handler = handler_cls(config=config, risk_manager=risk_manager, execution_engine=execution_engine)
            self.instances.append(StrategyInstance(config=config, handler=handler))
        self._queue: Optional["asyncio.Queue[Dict[str, Any]]"] = None
        self._task: Optional["asyncio.Task[None]"] = None

    def _match(self, event: Dict[str, Any], instance: StrategyInstance) -> bool:
        return (
            event.get("symbol") == instance.config.symbol
            and str(event.get("timeframe")) == str(instance.config.timeframe)
        )

    async def start(self, queue: "asyncio.Queue[Dict[str, Any]]") -> None:
        self._queue = queue
        if self._task and not self._task.done():
            return
        self._task = self.loop.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        if not self._queue:
            raise RuntimeError("Strategy manager was started without a queue")
        queue = self._queue
        while True:
            event = await queue.get()
            await self.dispatch_market_event(event)

    async def dispatch_market_event(self, event: Dict[str, Any]) -> None:
        for instance in self.instances:
            if not instance.enabled:
                continue
            if not self._match(event, instance):
                continue
            try:
                await instance.handler.on_market_data(event)
            except Exception:  # pragma: no cover - logging side effect
                self.logger.exception(
                    "Strategy '%s' failed processing market data", instance.config.name
                )

    async def dispatch_scheduled_event(self, event: Dict[str, Any]) -> None:
        for instance in self.instances:
            if not instance.enabled:
                continue
            if instance.config.timeframe != event.get("timeframe"):
                continue
            try:
                await instance.handler.on_scheduled_event(event)
            except Exception:  # pragma: no cover
                self.logger.exception(
                    "Strategy '%s' failed processing scheduled event", instance.config.name
                )

    def set_enabled(self, symbol: str, timeframe: str, name: str, enabled: bool) -> None:
        key = (symbol, timeframe, name)
        for instance in self.instances:
            if instance.key == key:
                instance.set_enabled(enabled)
                self.logger.info(
                    "%s strategy '%s' for %s %s",
                    "Enabled" if enabled else "Disabled",
                    name,
                    symbol,
                    timeframe,
                )
                return
        raise StrategyError(f"Strategy {name} not registered")


# Type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .risk import RiskManager
    from .execution import ExecutionEngine
