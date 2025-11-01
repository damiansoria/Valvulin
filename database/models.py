"""High level persistence helpers for optimisation and backtesting runs."""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, List

from database.utils import get_connection

LOGGER = logging.getLogger(__name__)


def _ensure_schema() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizations_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                method TEXT NOT NULL,
                expectancy REAL NOT NULL,
                profit_factor REAL NOT NULL,
                winrate REAL NOT NULL,
                drawdown REAL NOT NULL,
                sma_fast REAL NOT NULL,
                sma_slow REAL NOT NULL,
                rsi_period REAL NOT NULL,
                atr_mult_sl REAL NOT NULL,
                atr_mult_tp REAL NOT NULL,
                risk_per_trade REAL NOT NULL,
                extra JSON
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT,
                logic TEXT,
                capital REAL,
                metrics JSON
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS best_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                method TEXT NOT NULL,
                params JSON NOT NULL,
                metrics JSON NOT NULL
            );
            """
        )
        conn.commit()


def record_optimization(*, method: str, params: Dict[str, float], metrics: Dict[str, float]) -> None:
    """Persist an optimisation run inside ``optimizations_history``."""

    _ensure_schema()
    payload = {
        "created_at": dt.datetime.utcnow().isoformat(),
        "method": method,
        "expectancy": float(metrics.get("expectancy_r", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "winrate": float(metrics.get("winrate", 0.0)),
        "drawdown": float(metrics.get("max_drawdown", 0.0)),
        "sma_fast": float(params.get("sma_fast", 20)),
        "sma_slow": float(params.get("sma_slow", 80)),
        "rsi_period": float(params.get("rsi_period", 14)),
        "atr_mult_sl": float(params.get("atr_mult_sl", 1.5)),
        "atr_mult_tp": float(params.get("atr_mult_tp", 3.0)),
        "risk_per_trade": float(params.get("risk_per_trade", 1.0)),
        "extra": json.dumps(metrics),
    }
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO optimizations_history (
                created_at, method, expectancy, profit_factor, winrate, drawdown,
                sma_fast, sma_slow, rsi_period, atr_mult_sl, atr_mult_tp, risk_per_trade, extra
            ) VALUES (:created_at, :method, :expectancy, :profit_factor, :winrate, :drawdown,
                :sma_fast, :sma_slow, :rsi_period, :atr_mult_sl, :atr_mult_tp, :risk_per_trade, :extra)
            """,
            payload,
        )
        conn.commit()
    LOGGER.info("Registrada optimización %s con expectancy %.3f", method, payload["expectancy"])


def record_best_parameters(params: Dict[str, float], method: str, metrics: Dict[str, float]) -> None:
    """Store the best parameters snapshot for quick access."""

    _ensure_schema()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO best_parameters (created_at, method, params, metrics)
            VALUES (?, ?, ?, ?)
            """,
            (
                dt.datetime.utcnow().isoformat(),
                method,
                json.dumps(params),
                json.dumps(metrics),
            ),
        )
        conn.commit()


def fetch_recent_optimizations(*, limit: int = 3) -> List[Dict[str, float]]:
    """Return the most recent optimisation payloads ordered by ``created_at`` descending."""

    _ensure_schema()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT method, expectancy, profit_factor, winrate, drawdown,
                   sma_fast, sma_slow, rsi_period, atr_mult_sl, atr_mult_tp, risk_per_trade
            FROM optimizations_history
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
    records: List[Dict[str, float]] = []
    for row in rows:
        (
            method,
            expectancy,
            profit_factor,
            winrate,
            drawdown,
            sma_fast,
            sma_slow,
            rsi_period,
            atr_mult_sl,
            atr_mult_tp,
            risk_per_trade,
        ) = row
        records.append(
            {
                "method": method,
                "expectancy": expectancy,
                "profit_factor": profit_factor,
                "winrate": winrate,
                "drawdown": drawdown,
                "sma_fast": sma_fast,
                "sma_slow": sma_slow,
                "rsi_period": rsi_period,
                "atr_mult_sl": atr_mult_sl,
                "atr_mult_tp": atr_mult_tp,
                "risk_per_trade": risk_per_trade,
            }
        )
    return records


def record_backtest_run(
    *,
    symbol: str | None,
    logica: str,
    capital: float,
    metrics: Dict[str, float],
) -> None:
    """Persist a backtest execution for later auditing."""

    _ensure_schema()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO backtest_runs (created_at, symbol, logic, capital, metrics)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                dt.datetime.utcnow().isoformat(),
                symbol,
                logica,
                capital,
                json.dumps(metrics),
            ),
        )
        conn.commit()


__all__ = [
    "record_optimization",
    "fetch_recent_optimizations",
    "record_backtest_run",
    "record_best_parameters",
]
