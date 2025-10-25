"""Utilities for persisting and retrieving bot trades.

This module keeps track of historical trades and optionally open positions using
simple CSV backends.  The goal is to make it easy to persist the bot output in a
format that can be later re-used for analytics, dashboards or plotting.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"


@dataclass
class TradeEntry:
    """Represents a closed trade persisted in the ledger."""

    timestamp: datetime
    strategy: str
    r_multiple: float
    stop_loss: float
    take_profit: float
    notes: str = ""

    @classmethod
    def from_row(cls, row: dict) -> "TradeEntry":
        return cls(
            timestamp=datetime.fromisoformat(row["timestamp"]),
            strategy=row["strategy"],
            r_multiple=float(row["r_multiple"]),
            stop_loss=float(row["stop_loss"]),
            take_profit=float(row["take_profit"]),
            notes=row.get("notes", ""),
        )

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp.strftime(ISO_FORMAT),
            "strategy": self.strategy,
            "r_multiple": f"{self.r_multiple:.4f}",
            "stop_loss": f"{self.stop_loss:.4f}",
            "take_profit": f"{self.take_profit:.4f}",
            "notes": self.notes,
        }


@dataclass
class OpenTradeEntry:
    """Represents an open position tracked by the bot."""

    opened_at: datetime
    symbol: str
    strategy: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    notes: str = ""

    @classmethod
    def from_row(cls, row: dict) -> "OpenTradeEntry":
        return cls(
            opened_at=datetime.fromisoformat(row["opened_at"]),
            symbol=row["symbol"],
            strategy=row["strategy"],
            entry_price=float(row["entry_price"]),
            size=float(row["size"]),
            stop_loss=float(row["stop_loss"]),
            take_profit=float(row["take_profit"]),
            notes=row.get("notes", ""),
        )

    def to_row(self) -> dict:
        return {
            "opened_at": self.opened_at.strftime(ISO_FORMAT),
            "symbol": self.symbol,
            "strategy": self.strategy,
            "entry_price": f"{self.entry_price:.4f}",
            "size": f"{self.size:.4f}",
            "stop_loss": f"{self.stop_loss:.4f}",
            "take_profit": f"{self.take_profit:.4f}",
            "notes": self.notes,
        }


class TradeLogger:
    """Helper responsible for storing trades and open positions in CSV files."""

    def __init__(
        self,
        trades_path: Path | str = Path("data/trades.csv"),
        open_trades_path: Path | str = Path("data/open_trades.csv"),
    ) -> None:
        self.trades_path = Path(trades_path)
        self.open_trades_path = Path(open_trades_path)
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        self.open_trades_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Closed trades helpers
    # ------------------------------------------------------------------
    def log_trade(
        self,
        strategy: str,
        r_multiple: float,
        stop_loss: float,
        take_profit: float,
        notes: str = "",
        timestamp: Optional[datetime] = None,
    ) -> TradeEntry:
        """Persist a closed trade to disk and return the stored entry."""

        entry = TradeEntry(
            timestamp=timestamp or datetime.utcnow(),
            strategy=strategy,
            r_multiple=r_multiple,
            stop_loss=stop_loss,
            take_profit=take_profit,
            notes=notes,
        )
        write_header = not self.trades_path.exists()
        with self.trades_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "timestamp",
                    "strategy",
                    "r_multiple",
                    "stop_loss",
                    "take_profit",
                    "notes",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(entry.to_row())
        return entry

    def load_trades(self) -> List[TradeEntry]:
        if not self.trades_path.exists():
            return []
        with self.trades_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [TradeEntry.from_row(row) for row in reader]

    def save_trades(self, trades: Iterable[TradeEntry]) -> None:
        with self.trades_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "timestamp",
                    "strategy",
                    "r_multiple",
                    "stop_loss",
                    "take_profit",
                    "notes",
                ],
            )
            writer.writeheader()
            for trade in trades:
                writer.writerow(trade.to_row())

    # ------------------------------------------------------------------
    # Open positions helpers
    # ------------------------------------------------------------------
    def log_open_trade(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        notes: str = "",
        opened_at: Optional[datetime] = None,
    ) -> OpenTradeEntry:
        """Append a new open trade to the open-trades ledger."""

        entry = OpenTradeEntry(
            opened_at=opened_at or datetime.utcnow(),
            symbol=symbol,
            strategy=strategy,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            notes=notes,
        )
        write_header = not self.open_trades_path.exists()
        with self.open_trades_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "opened_at",
                    "symbol",
                    "strategy",
                    "entry_price",
                    "size",
                    "stop_loss",
                    "take_profit",
                    "notes",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(entry.to_row())
        return entry

    def load_open_trades(self) -> List[OpenTradeEntry]:
        if not self.open_trades_path.exists():
            return []
        with self.open_trades_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [OpenTradeEntry.from_row(row) for row in reader]

    def overwrite_open_trades(self, trades: Iterable[OpenTradeEntry]) -> None:
        with self.open_trades_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "opened_at",
                    "symbol",
                    "strategy",
                    "entry_price",
                    "size",
                    "stop_loss",
                    "take_profit",
                    "notes",
                ],
            )
            writer.writeheader()
            for trade in trades:
                writer.writerow(trade.to_row())


__all__ = [
    "TradeEntry",
    "OpenTradeEntry",
    "TradeLogger",
]
