"""Structured logging utilities with rotating file handlers."""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "valvulin.log"


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for easier ingestion."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited docstring
        log_record: Dict[str, Any] = {
            "level": record.levelname,
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        if record.__dict__:
            extras = {
                key: value
                for key, value in record.__dict__.items()
                if key not in logging.LogRecord.__dict__
            }
            log_record.update(extras)
        return json.dumps(log_record)


def setup_logging(
    level: int = logging.INFO,
    log_file: Path = DEFAULT_LOG_FILE,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> None:
    """Configure application-wide logging with JSON formatting and rotation."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to prevent duplicate logs during reloads/tests.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = JsonFormatter()

    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger."""

    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger", "JsonFormatter"]
