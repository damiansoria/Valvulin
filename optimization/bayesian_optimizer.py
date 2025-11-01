"""Bayesian optimisation utilities built on top of scikit-optimize."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from analytics.backtest_visual import run_backtest

try:  # Optional dependency used for Bayesian search
    from skopt import gp_minimize
    from skopt.space import Integer, Real
except ImportError:  # pragma: no cover - optional dependency
    gp_minimize = None  # type: ignore[assignment]
    Integer = Real = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BayesianResult:
    """Container for a Bayesian optimisation output."""

    params: Dict[str, float]
    history: List[Dict[str, float]]
    metrics: Dict[str, float]
    method: str = "Bayesian"


BOUNDS: Dict[str, Tuple[float, float]] = {
    "sma_fast": (10, 50),
    "sma_slow": (60, 150),
    "rsi_period": (10, 30),
    "atr_mult_sl": (1.0, 3.0),
    "atr_mult_tp": (2.0, 6.0),
    "risk_per_trade": (0.5, 2.0),
}


def _build_space() -> List[Integer | Real]:
    """Return a skopt compatible search space definition."""

    if Integer is None or Real is None:
        raise RuntimeError(
            "scikit-optimize no está instalado. Ejecuta `pip install scikit-optimize` para usar la optimización bayesiana."
        )

    space: List[Integer | Real] = [
        Integer(BOUNDS["sma_fast"][0], BOUNDS["sma_fast"][1], name="sma_fast"),
        Integer(BOUNDS["sma_slow"][0], BOUNDS["sma_slow"][1], name="sma_slow"),
        Integer(BOUNDS["rsi_period"][0], BOUNDS["rsi_period"][1], name="rsi_period"),
        Real(BOUNDS["atr_mult_sl"][0], BOUNDS["atr_mult_sl"][1], name="atr_mult_sl"),
        Real(BOUNDS["atr_mult_tp"][0], BOUNDS["atr_mult_tp"][1], name="atr_mult_tp"),
        Real(BOUNDS["risk_per_trade"][0], BOUNDS["risk_per_trade"][1], name="risk_per_trade"),
    ]
    return space


def _evaluate(
    params: Iterable[float],
    *,
    data: pd.DataFrame,
    strategies: Iterable[str],
    capital: float,
    logica: str,
    symbol: str | None,
) -> Tuple[float, Dict[str, float]]:
    """Objective function for Bayesian optimisation."""

    values = list(params)
    mapped = {
        "sma_fast": int(values[0]),
        "sma_slow": int(values[1]),
        "rsi_period": int(values[2]),
        "atr_mult_sl": float(values[3]),
        "atr_mult_tp": float(values[4]),
        "risk_per_trade": float(values[5]),
    }

    strategy_params = {
        "SMA Crossover": {"fast": mapped["sma_fast"], "slow": mapped["sma_slow"]},
        "RSI": {"period": mapped["rsi_period"]},
    }

    result = run_backtest(
        data=data,
        strategies=list(strategies),
        params=strategy_params,
        capital_inicial=capital,
        riesgo_por_trade=mapped["risk_per_trade"],
        sl_ratio=mapped["atr_mult_sl"],
        tp_ratio=mapped["atr_mult_tp"],
        logica=logica,
        symbol=symbol,
    )
    metrics = result.metrics
    expectancy = float(metrics.get("Expectancy R", metrics.get("Expectancy", 0.0)))
    profit_factor = float(metrics.get("Profit Factor", 0.0))
    metrics_payload = {
        "expectancy_r": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": float(metrics.get("Max Drawdown %", 0.0)),
        "winrate": float(metrics.get("Winrate %", 0.0)),
    }
    LOGGER.debug("Bayesian eval params=%s metrics=%s", mapped, metrics_payload)
    return -expectancy, metrics_payload


def run_bayesian_optimization(
    *,
    data: pd.DataFrame,
    strategies: Iterable[str],
    capital: float,
    logica: str,
    symbol: str | None,
    n_calls: int = 40,
    random_state: int = 42,
    initial_points: List[Dict[str, float]] | None = None,
) -> BayesianResult:
    """Execute the Bayesian optimiser returning the best configuration found."""

    if gp_minimize is None:
        raise RuntimeError(
            "scikit-optimize es requerido para la optimización bayesiana."
        )

    space = _build_space()

    history: List[Dict[str, float]] = []

    def objective(values: List[float]) -> float:
        score, metrics = _evaluate(
            values,
            data=data,
            strategies=strategies,
            capital=capital,
            logica=logica,
            symbol=symbol,
        )
        payload = {
            "sma_fast": float(values[0]),
            "sma_slow": float(values[1]),
            "rsi_period": float(values[2]),
            "atr_mult_sl": float(values[3]),
            "atr_mult_tp": float(values[4]),
            "risk_per_trade": float(values[5]),
            **metrics,
        }
        history.append(payload)
        return score

    x0 = None
    y0 = None
    if initial_points:
        x0 = [
            [
                point["sma_fast"],
                point["sma_slow"],
                point["rsi_period"],
                point["atr_mult_sl"],
                point["atr_mult_tp"],
                point["risk_per_trade"],
            ]
            for point in initial_points
        ]
        y0 = []
        for point in initial_points:
            score, metrics = _evaluate(
                [
                    point["sma_fast"],
                    point["sma_slow"],
                    point["rsi_period"],
                    point["atr_mult_sl"],
                    point["atr_mult_tp"],
                    point["risk_per_trade"],
                ],
                data=data,
                strategies=strategies,
                capital=capital,
                logica=logica,
                symbol=symbol,
            )
            history.append({**point, **metrics})
            y0.append(score)

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=random_state,
        n_initial_points=len(x0) if x0 else 10,
        x0=x0,
        y0=y0,
    )

    best_params = {
        "sma_fast": int(result.x[0]),
        "sma_slow": int(result.x[1]),
        "rsi_period": int(result.x[2]),
        "atr_mult_sl": float(result.x[3]),
        "atr_mult_tp": float(result.x[4]),
        "risk_per_trade": float(result.x[5]),
    }

    _, metrics_payload = _evaluate(
        result.x,
        data=data,
        strategies=strategies,
        capital=capital,
        logica=logica,
        symbol=symbol,
    )

    return BayesianResult(
        params=best_params,
        history=history,
        metrics=metrics_payload,
    )


__all__ = ["run_bayesian_optimization", "BayesianResult", "BOUNDS"]
