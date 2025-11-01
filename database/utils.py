"""Utilities to interact with the SQLite storage used by Valvulin."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

DEFAULT_DB_PATH = Path("results/valvulin.db")


def _resolve_path(path: str | Path | None = None) -> Path:
    resolved = Path(path) if path else DEFAULT_DB_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@contextmanager
def get_connection(path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager returning a SQLite connection with foreign keys enabled."""

    database_path = _resolve_path(path)
    conn = sqlite3.connect(database_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


__all__ = ["get_connection", "DEFAULT_DB_PATH"]
