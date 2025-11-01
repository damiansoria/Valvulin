"""Simple genetic algorithm wrapper for parameter search."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from analytics.backtest_visual import run_backtest

LOGGER = logging.getLogger(__name__)


BOUNDS: Dict[str, Tuple[float, float]] = {
    "sma_fast": (10, 50),
    "sma_slow": (60, 150),
    "rsi_period": (10, 30),
    "atr_mult_sl": (1.0, 3.0),
    "atr_mult_tp": (2.0, 6.0),
    "risk_per_trade": (0.5, 2.0),
}

INTEGER_PARAMS = {"sma_fast", "sma_slow", "rsi_period"}


@dataclass(slots=True)
class GeneticResult:
    """Container with the best GA output and evaluation history."""

    params: Dict[str, float]
    history: List[Dict[str, float]]
    metrics: Dict[str, float]
    method: str = "Genetic"


def _random_param(name: str) -> float:
    low, high = BOUNDS[name]
    if name in INTEGER_PARAMS:
        return float(random.randint(int(low), int(high)))
    return random.uniform(low, high)


def _initial_population(size: int) -> List[Dict[str, float]]:
    return [
        {name: _random_param(name) for name in BOUNDS}
        for _ in range(size)
    ]


def _mutate(individual: Dict[str, float], rate: float) -> Dict[str, float]:
    mutated = individual.copy()
    for name in BOUNDS:
        if random.random() < rate:
            mutated[name] = _random_param(name)
    return mutated


def _crossover(parent_a: Dict[str, float], parent_b: Dict[str, float]) -> Dict[str, float]:
    child = {}
    for name in BOUNDS:
        if random.random() < 0.5:
            child[name] = parent_a[name]
        else:
            child[name] = parent_b[name]
    return child


def _evaluate(
    params: Dict[str, float],
    *,
    data: pd.DataFrame,
    strategies: Iterable[str],
    capital: float,
    logica: str,
    symbol: str | None,
) -> Tuple[float, Dict[str, float]]:
    strategy_params = {
        "SMA Crossover": {
            "fast": int(params["sma_fast"]),
            "slow": int(params["sma_slow"]),
        },
        "RSI": {"period": int(params["rsi_period"])},
    }

    result = run_backtest(
        data=data,
        strategies=list(strategies),
        params=strategy_params,
        capital_inicial=capital,
        riesgo_por_trade=params["risk_per_trade"],
        sl_ratio=params["atr_mult_sl"],
        tp_ratio=params["atr_mult_tp"],
        logica=logica,
        symbol=symbol,
    )
    metrics = result.metrics
    expectancy = float(metrics.get("Expectancy R", metrics.get("Expectancy", 0.0)))
    profit_factor = float(metrics.get("Profit Factor", 0.0))
    payload = {
        "expectancy_r": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": float(metrics.get("Max Drawdown %", 0.0)),
        "winrate": float(metrics.get("Winrate %", 0.0)),
    }
    LOGGER.debug("Genetic eval params=%s metrics=%s", params, payload)
    return expectancy, payload


def run_genetic_optimization(
    *,
    data: pd.DataFrame,
    strategies: Iterable[str],
    capital: float,
    logica: str,
    symbol: str | None,
    population_size: int = 30,
    generations: int = 25,
    elite_ratio: float = 0.2,
    mutation_rate: float = 0.2,
    seeded_individuals: List[Dict[str, float]] | None = None,
) -> GeneticResult:
    """Execute a lightweight GA tailored for Valvulin parameters."""

    population = _initial_population(population_size)
    if seeded_individuals:
        for seed in seeded_individuals:
            population[random.randrange(population_size)] = seed.copy()

    history: List[Dict[str, float]] = []
    best_params: Dict[str, float] = population[0]
    best_metrics = {
        "expectancy_r": -float("inf"),
        "profit_factor": 0.0,
        "max_drawdown": float("inf"),
        "winrate": 0.0,
    }

    elite_count = max(1, int(population_size * elite_ratio))

    for generation in range(generations):
        scored: List[Tuple[float, Dict[str, float], Dict[str, float]]] = []
        for individual in population:
            score, metrics = _evaluate(
                individual,
                data=data,
                strategies=strategies,
                capital=capital,
                logica=logica,
                symbol=symbol,
            )
            scored.append((score, individual, metrics))
            history.append({**individual, **metrics, "generation": generation})

        scored.sort(key=lambda item: item[0], reverse=True)
        elite = [item[1] for item in scored[:elite_count]]
        if scored[0][0] > best_metrics["expectancy_r"]:
            best_params = scored[0][1].copy()
            best_metrics = scored[0][2]

        new_population: List[Dict[str, float]] = elite.copy()
        while len(new_population) < population_size:
            parent_a = random.choice(elite)
            parent_b = random.choice(population)
            child = _crossover(parent_a, parent_b)
            child = _mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    return GeneticResult(
        params=best_params,
        history=history,
        metrics=best_metrics,
    )


__all__ = ["run_genetic_optimization", "GeneticResult", "BOUNDS"]
