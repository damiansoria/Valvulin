"""Hybrid optimisation pipeline mixing Bayesian and Genetic approaches."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from database import models
from optimization.bayesian_optimizer import (
    BayesianResult,
    run_bayesian_optimization,
)
from optimization.genetic_optimizer import GeneticResult, run_genetic_optimization

LOGGER = logging.getLogger(__name__)

BEST_PARAMS_PATH = Path("results/best_params.json")


@dataclass(slots=True)
class HybridResult:
    """Container describing the outcome of the hybrid optimisation."""

    best_params: Dict[str, float]
    method: str
    bayesian: BayesianResult
    genetic: GeneticResult
    history: pd.DataFrame


DEFAULT_STRATEGIES = ("SMA Crossover", "RSI")


def choose_best_optimizer(
    results_bayes: Dict[str, float], results_genetic: Dict[str, float]
) -> Tuple[Dict[str, float], str]:
    """Return the result with the highest expectancy, breaking ties by profit factor."""

    if results_bayes.get("expectancy_r", 0.0) > results_genetic.get("expectancy_r", 0.0):
        return results_bayes, "Bayesian"
    if results_bayes.get("expectancy_r", 0.0) == results_genetic.get("expectancy_r", 0.0):
        if results_bayes.get("profit_factor", 0.0) >= results_genetic.get("profit_factor", 0.0):
            return results_bayes, "Bayesian"
    return results_genetic, "Genetic"


def _collect_weighted_seeds(limit: int = 3) -> List[Dict[str, float]]:
    """Extract the best recent parameter sets to seed the optimisers."""

    history = models.fetch_recent_optimizations(limit=limit)
    if not history:
        return []

    weights = list(range(len(history), 0, -1))
    total = sum(weights)
    seeds: List[Dict[str, float]] = []
    for weight, record in zip(weights, history):
        params = {
            "sma_fast": float(record.get("sma_fast", 20)),
            "sma_slow": float(record.get("sma_slow", 80)),
            "rsi_period": float(record.get("rsi_period", 14)),
            "atr_mult_sl": float(record.get("atr_mult_sl", 1.5)),
            "atr_mult_tp": float(record.get("atr_mult_tp", 3.0)),
            "risk_per_trade": float(record.get("risk_per_trade", 1.0)),
        }
        # Weighted jitter to favour recent configurations without collapsing diversity.
        factor = weight / total
        for key in params:
            params[key] = params[key] * (1 + (0.05 * factor))
        seeds.append(params)
    return seeds


def _persist_best(best: Dict[str, float], method: str, metrics: Dict[str, float]) -> None:
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as handler:
        json.dump({"method": method, "params": best, "metrics": metrics}, handler, indent=2)

    models.record_best_parameters(best, method, metrics)


def run_hybrid_optimization(
    *,
    data: pd.DataFrame,
    capital_inicial: float,
    riesgo_por_trade: float,
    logica: str,
    symbol: str | None,
    strategies: Iterable[str] | None = None,
    bayes_calls: int = 40,
    ga_generations: int = 25,
    ga_population: int = 30,
) -> HybridResult:
    """Run both optimisers, compare their outputs and persist the best configuration."""

    if data is None or data.empty:
        raise ValueError("Se requiere un DataFrame con datos para optimizar.")

    strategies = tuple(strategies or DEFAULT_STRATEGIES)

    seeds = _collect_weighted_seeds()
    LOGGER.info("Weighted seeding generated %s candidates", len(seeds))

    bayesian_result: BayesianResult
    try:
        bayesian_result = run_bayesian_optimization(
            data=data,
            strategies=strategies,
            capital=capital_inicial,
            logica=logica,
            symbol=symbol,
            n_calls=bayes_calls,
            random_state=42,
            initial_points=seeds,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.exception("Bayesian optimisation failed: %s", exc)
        bayesian_result = BayesianResult(
            params=seeds[0] if seeds else {
                "sma_fast": 20,
                "sma_slow": 80,
                "rsi_period": 14,
                "atr_mult_sl": 1.5,
                "atr_mult_tp": 3.0,
                "risk_per_trade": max(riesgo_por_trade, 0.5),
            },
            history=[],
            metrics={"expectancy_r": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "winrate": 0.0},
        )

    genetic_result = run_genetic_optimization(
        data=data,
        strategies=strategies,
        capital=capital_inicial,
        logica=logica,
        symbol=symbol,
        population_size=ga_population,
        generations=ga_generations,
        seeded_individuals=seeds,
    )

    best_payload, method = choose_best_optimizer(
        bayesian_result.metrics,
        genetic_result.metrics,
    )

    best_params = bayesian_result.params if method == "Bayesian" else genetic_result.params

    history_frames: List[pd.DataFrame] = []
    if bayesian_result.history:
        history_frames.append(pd.DataFrame(bayesian_result.history).assign(method="Bayesian"))
    if genetic_result.history:
        history_frames.append(pd.DataFrame(genetic_result.history).assign(method="Genetic"))
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()

    models.record_optimization(
        method=method,
        params=best_params,
        metrics=best_payload,
    )
    _persist_best(best_params, method, best_payload)

    return HybridResult(
        best_params=best_params,
        method=method,
        bayesian=bayesian_result,
        genetic=genetic_result,
        history=history_df,
    )


__all__ = ["run_hybrid_optimization", "choose_best_optimizer", "HybridResult"]
