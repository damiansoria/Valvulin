"""Order execution helpers."""
from __future__ import annotations

import logging
from typing import Any, Dict


class ExecutionEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger("valvulin.execution")

    async def submit(self, order: Dict[str, Any]) -> None:
        self.logger.info("Submitting order: %s", order)
