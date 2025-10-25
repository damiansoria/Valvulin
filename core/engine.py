"""Main trading engine orchestration.

The :class:`TradingEngine` wires together configuration loading, data feeds,
strategy scheduling and order execution.  It is responsible for bootstrapping
the asynchronous services and acting as the integration point between the
``core`` and ``execution`` packages.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .config import (
    BotConfig,
    SchedulerInterval,
    apply_cli_overrides,
    create_parser,
    load_config,
)
from .data_feed import DataFeedManager
from .execution import ExecutionEngine
from .risk import RiskManager
from .scheduler import AsyncScheduler
from .strategy import StrategyManager

LOGGER = logging.getLogger("valvulin.engine")


class TradingEngine:
    """Coordinates data feeds, strategies, risk management and execution."""

    def __init__(
        self, config: BotConfig, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        self.config = config
        self.loop = loop or asyncio.get_event_loop()
        self.logger = LOGGER
        self.queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self.risk_manager = RiskManager(config.risk)
        self.execution_engine = ExecutionEngine(config.execution)
        self.strategy_manager = StrategyManager(
            config.strategies,
            risk_manager=self.risk_manager,
            execution_engine=self.execution_engine,
            loop=self.loop,
        )
        self.data_feed_manager = DataFeedManager(
            config.data_feeds,
            queue=self.queue,
            loop=self.loop,
        )
        self.scheduler = AsyncScheduler(
            loop=self.loop, logger=logging.getLogger("valvulin.scheduler")
        )
        self._started = False
        self._register_scheduler_jobs(config)

    def _register_scheduler_jobs(self, config: BotConfig) -> None:
        """Register all interval jobs defined in the configuration file."""

        for interval in config.scheduler.intervals:
            self._register_interval(interval)

    def _register_interval(self, interval: SchedulerInterval) -> None:
        """Create a scheduled callback that dispatches timeframe events."""

        async def callback(timeframe: str = interval.timeframe) -> None:
            event = {
                "timeframe": timeframe,
                "source": "scheduler",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.strategy_manager.dispatch_scheduled_event(event)

        name = f"schedule:{interval.timeframe}:{interval.seconds}"
        self.scheduler.add_job(
            name=name,
            interval=float(interval.seconds),
            callback=callback,
            warmup=float(interval.warmup_seconds or 0),
        )
        self.logger.debug("Registered scheduler job %s", name)

    async def start(self) -> None:
        """Start all sub-systems once, ensuring idempotency."""

        if self._started:
            return
        self.logger.info("Starting trading engine")
        await self.strategy_manager.start(self.queue)
        await self.data_feed_manager.start()
        await self.scheduler.start()
        self._started = True

    async def stop(self) -> None:
        """Stop all sub-systems in reverse order of their dependencies."""

        if not self._started:
            return
        self.logger.info("Stopping trading engine")
        await self.scheduler.stop()
        await self.data_feed_manager.stop()
        await self.strategy_manager.stop()
        self._started = False

    async def run_forever(self) -> None:
        """Run the engine until a termination signal is received."""

        await self.start()
        stop_event = asyncio.Event()

        def _shutdown_handler(*_: int) -> None:
            self.logger.info("Shutdown signal received")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self.loop.add_signal_handler(sig, _shutdown_handler)
            except NotImplementedError:  # pragma: no cover - Windows
                signal.signal(sig, lambda *_: stop_event.set())

        try:
            await stop_event.wait()
        finally:
            await self.stop()

    def set_strategy_enabled(
        self, symbol: str, timeframe: str, name: str, enabled: bool
    ) -> None:
        """Toggle an individual strategy on or off at runtime."""

        self.strategy_manager.set_enabled(symbol, timeframe, name, enabled)
        self.config.set_strategy_enabled(symbol, timeframe, name, enabled)


async def _async_main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    engine = TradingEngine(config)
    await engine.run_forever()


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.info("Interrupted by user")


if __name__ == "__main__":  # pragma: no cover
    main()
