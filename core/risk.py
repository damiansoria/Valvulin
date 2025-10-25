"""Risk management helpers."""
from __future__ import annotations

import logging
from typing import Any, Dict


class RiskManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger("valvulin.risk")

    def assess(self, signal: Dict[str, Any]) -> bool:
        """Return True when the proposed trade is allowed."""
        max_position = self.config.get("max_position")
        if max_position is None:
            return True
        size = signal.get("size", 0)
        if abs(size) > max_position:
            self.logger.warning(
                "Rejected trade because size %s exceeds max_position %s", size, max_position
            )
            return False
        return True
