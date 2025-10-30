"""Persistence helpers for Streamlit backtests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from analytics.csv_normalization import normalize_trade_dataframe

RESULTS_DIR = Path("results")


def _ensure_results_dir() -> Path:
    """Create the results directory if it does not exist."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _serialise_parameters(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a JSON-serialisable copy of the strategy parameters."""

    if not params:
        return {}
    serialised: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, dict):
            serialised[key] = _serialise_parameters(value)  # type: ignore[assignment]
        elif isinstance(value, (list, tuple)):
            serialised[key] = [
                item if isinstance(item, (int, float, str, bool)) else str(item)
                for item in value
            ]
        elif isinstance(value, (int, float, str, bool)) or value is None:
            serialised[key] = value
        else:
            serialised[key] = str(value)
    return serialised


@dataclass(slots=True)
class BacktestRunSummary:
    """Lightweight descriptor for a stored backtest run."""

    run_id: str
    path: Path
    metadata: Dict[str, Any]
    trades_path: Path
    metrics_path: Optional[Path]

    def label(self) -> str:
        """Return a human readable label for dropdowns."""

        executed_at = self.metadata.get("executed_at")
        timestamp_label = self.run_id
        if isinstance(executed_at, str):
            try:
                parsed = datetime.fromisoformat(executed_at)
                timestamp_label = parsed.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                timestamp_label = executed_at
        symbol = self.metadata.get("symbol", "?")
        strategy = self.metadata.get("strategy_label") or self.metadata.get("strategy_name")
        strategy_label = strategy if isinstance(strategy, str) else "Estrategia"
        return f"{timestamp_label} · {symbol} · {strategy_label}"


def save_backtest_run(
    result: "StrategyResult",
    *,
    symbol: str,
    interval: str,
    strategies: Sequence[str],
    params: Dict[str, Any],
    capital_inicial: float,
    riesgo_por_trade: float,
    sl_ratio: float,
    tp_ratio: float,
    logica: str,
) -> BacktestRunSummary:
    """Persist the result of a backtest run to ``results/``."""

    _ensure_results_dir()
    executed_at = datetime.utcnow()
    run_id = executed_at.strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / run_id
    counter = 1
    while run_dir.exists():
        counter += 1
        run_dir = RESULTS_DIR / f"{run_id}-{counter}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trades_df = result.trades.copy()
    if not trades_df.empty:
        trades_df, _ = normalize_trade_dataframe(trades_df)
        if "symbol" not in trades_df.columns:
            trades_df["symbol"] = symbol
        else:
            trades_df["symbol"] = trades_df["symbol"].fillna(symbol)
        strategy_label = "+".join(strategies)
        if "strategy" not in trades_df.columns:
            trades_df["strategy"] = strategy_label
        else:
            trades_df["strategy"] = trades_df["strategy"].fillna(strategy_label)

    trades_path = run_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)

    metrics_records: Iterable[tuple[str, Any]]
    metrics_records = result.metrics.items() if result.metrics else []
    metrics_df = pd.DataFrame(metrics_records, columns=["metric", "value"])
    metrics_path: Optional[Path] = None
    if not metrics_df.empty:
        metrics_path = run_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

    final_equity = capital_inicial
    if not result.equity_curve.empty:
        final_equity = float(result.equity_curve.iloc[-1])
    elif "capital_final" in trades_df.columns and not trades_df.empty:
        final_equity = float(trades_df["capital_final"].iloc[-1])

    metadata: Dict[str, Any] = {
        "run_id": run_dir.name,
        "executed_at": executed_at.isoformat(),
        "symbol": symbol,
        "interval": interval,
        "strategy_name": list(strategies),
        "strategy_label": "+".join(strategies),
        "parameters": _serialise_parameters(params),
        "capital_inicial": float(capital_inicial),
        "capital_final": float(final_equity),
        "riesgo_por_trade": float(riesgo_por_trade),
        "sl_ratio": float(sl_ratio),
        "tp_ratio": float(tp_ratio),
        "logica": logica,
        "trades_csv": trades_path.name,
    }
    if metrics_path is not None:
        metadata["metrics_csv"] = metrics_path.name

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return BacktestRunSummary(
        run_id=run_dir.name,
        path=run_dir,
        metadata=metadata,
        trades_path=trades_path,
        metrics_path=metrics_path,
    )


def list_backtest_runs() -> List[BacktestRunSummary]:
    """Return the available stored backtests sorted by newest first."""

    if not RESULTS_DIR.exists():
        return []

    summaries: List[BacktestRunSummary] = []
    for metadata_path in sorted(RESULTS_DIR.glob("*/metadata.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        run_dir = metadata_path.parent
        trades_path = run_dir / metadata.get("trades_csv", "trades.csv")
        metrics_name = metadata.get("metrics_csv")
        metrics_path = run_dir / metrics_name if metrics_name else None
        if not trades_path.exists():
            continue
        summaries.append(
            BacktestRunSummary(
                run_id=run_dir.name,
                path=run_dir,
                metadata=metadata,
                trades_path=trades_path,
                metrics_path=metrics_path if metrics_path and metrics_path.exists() else None,
            )
        )

    def _sort_key(item: BacktestRunSummary) -> tuple[int, str]:
        executed_at = item.metadata.get("executed_at")
        if isinstance(executed_at, str):
            try:
                ts = datetime.fromisoformat(executed_at)
                return (0, ts.isoformat())
            except ValueError:
                return (1, executed_at)
        return (2, item.run_id)

    summaries.sort(key=_sort_key, reverse=True)
    return summaries


def load_backtest_run(
    summary: BacktestRunSummary,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load the trades and metrics associated with *summary*."""

    trades_df = pd.read_csv(summary.trades_path)
    if summary.metrics_path and summary.metrics_path.exists():
        metrics_df = pd.read_csv(summary.metrics_path)
    else:
        metrics_df = None
    return trades_df, metrics_df


__all__ = [
    "BacktestRunSummary",
    "list_backtest_runs",
    "load_backtest_run",
    "save_backtest_run",
]
