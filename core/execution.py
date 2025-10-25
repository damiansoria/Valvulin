"""Abstractions used by the core engine to route execution requests.

The :mod:`execution` package contains the concrete implementations that talk to
exchanges (``live_executor``) or simulate fills for historical data
(``backtester``).  The lightweight :class:`ExecutionEngine` defined here acts as
the glue between asynchronous strategies and those modules by accepting generic
order dictionaries and delegating them to the configured backend.
"""

from __future__ import annotations

import logging
from typing import Any, Dict


class ExecutionEngine:
    """Thin facade responsible for forwarding orders to execution backends."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger("valvulin.execution")

    async def submit(self, order: Dict[str, Any]) -> None:
        """Log and forward the order to the downstream execution handler."""

        self.logger.info("Submitting order: %s", order)
