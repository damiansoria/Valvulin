"""Example strategies used for testing the engine."""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.strategy import BaseStrategy


class EchoStrategy(BaseStrategy):
    """A simple strategy that logs incoming events."""

    async def on_market_data(self, event: Dict[str, Any]) -> None:
        if not self.risk_manager.assess({"size": event.get("volume", 0)}):
            self.logger.info("Risk prevented trade for %s", event["symbol"])
            return
        await self.execution_engine.submit(
            {
                "symbol": event["symbol"],
                "timeframe": event.get("timeframe"),
                "action": "log",
                "metadata": event,
            }
        )

    async def on_scheduled_event(self, event: Dict[str, Any]) -> None:
        logging.getLogger("valvulin.strategy.echo").info(
            "Scheduled check for timeframe %s", event.get("timeframe")
        )
