"""Advanced parameter optimization utilities for Valvulin."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from analytics.backtest_visual import run_backtest

try:  # Optional dependency used for the genetic algorithm mode
    from deap import base, creator, tools
except ImportError:  # pragma: no cover - handled at runtime
    base = creator = tools = None  # type: ignore[assignment]

try:  # Optional dependency used for the Bayesian optimisation mode
    from skopt import gp_minimize
    from skopt.space import Integer, Real
except ImportError:  # pragma: no cover - handled at runtime
    gp_minimize = None  # type: ignore[assignment]
    Integer = Real = None  # type: ignore[assignment]


PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ema_fast": (10, 30),
    "ema_slow": (50, 150),
    "rsi_period": (7, 21),
    "atr_mult_sl": (1.5, 3.0),
    "atr_mult_tp": (3.0, 6.0),
}

RESULTS_PATH = Path("results/optimizer_advanced.csv")


@dataclass(slots=True)
class Evaluation:
    """Container for optimisation metrics."""

    params: Dict[str, float]
    expectancy_R: float
    max_drawdown: float
    profit_factor: float
    rr_effective: float
    fitness: float
    mark_as_best: bool
    meets_target: bool


class AdvancedOptimizer:
    """Runs advanced optimisation workflows using GA or Bayesian search."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        strategies: Sequence[str] | None = None,
        capital_inicial: float = 1_000.0,
        riesgo_por_trade: float = 1.0,
        logica: str = "AND",
        symbol: str | None = None,
    ) -> None:
        if data is None or data.empty:
            raise ValueError("El dataset proporcionado está vacío.")
        self.data = data.copy()
        self.strategies: Sequence[str] = strategies or ("SMA Crossover", "RSI")
        self.capital_inicial = float(capital_inicial)
        self.riesgo_por_trade = float(riesgo_por_trade)
        self.logica = logica
        self.symbol = symbol

        self._cache: Dict[Tuple[float, ...], Evaluation] = {}
        self._records: List[Dict[str, Any]] = []
        self._recorded_keys: set[Tuple[Tuple[float, ...], str]] = set()

        if not set(self.strategies).issuperset({"SMA Crossover", "RSI"}):
            raise ValueError(
                "Las estrategias deben incluir 'SMA Crossover' y 'RSI' para la optimización avanzada."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(
        self,
        mode: str,
        *,
        population_size: int = 30,
        generations: int = 20,
        elite_ratio: float = 0.2,
        n_calls: int = 60,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Execute the optimisation pipeline and return all evaluated results."""

        normalized_mode = mode.lower()
        if "gen" in normalized_mode:
            self._run_genetic(
                population_size=population_size,
                generations=generations,
                elite_ratio=elite_ratio,
                random_state=random_state,
            )
        elif "bayes" in normalized_mode:
            self._run_bayesian(n_calls=n_calls, random_state=random_state)
        else:  # pragma: no cover - defensive programming
            raise ValueError("Modo de optimización no soportado. Usa 'genetico' o 'bayesiano'.")

        return self._results_dataframe()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _results_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "ema_fast",
                    "ema_slow",
                    "rsi_period",
                    "atr_mult_sl",
                    "atr_mult_tp",
                    "expectancy_R",
                    "max_drawdown",
                    "profit_factor",
                    "rr_effective",
                    "fitness",
                    "mark_as_best",
                    "meets_target",
                    "method",
                ]
            )

        df = pd.DataFrame(self._records)
        df.sort_values(
            by=["expectancy_R", "max_drawdown"], ascending=[False, True], inplace=True
        )

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.head(10).to_csv(RESULTS_PATH, index=False)
        return df

    def _store_record(self, key: Tuple[float, ...], method: str, evaluation: Evaluation) -> None:
        record_key = (key, method)
        if record_key in self._recorded_keys:
            return
        self._recorded_keys.add(record_key)

        row = {
            **{name: evaluation.params[name] for name in PARAM_BOUNDS},
            "expectancy_R": evaluation.expectancy_R,
            "max_drawdown": evaluation.max_drawdown,
            "profit_factor": evaluation.profit_factor,
            "rr_effective": evaluation.rr_effective,
            "fitness": evaluation.fitness,
            "mark_as_best": evaluation.mark_as_best,
            "meets_target": evaluation.meets_target,
            "method": method,
        }
        self._records.append(row)

    @staticmethod
    def _make_key(params: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(float(params[name]) for name in PARAM_BOUNDS)

    @staticmethod
    def _decode_individual(individual: Iterable[float]) -> Dict[str, float]:
        decoded: Dict[str, float] = {}
        for gene, (name, bounds) in zip(individual, PARAM_BOUNDS.items()):
            low, high = bounds
            value = low + (high - low) * float(gene)
            if name in {"ema_fast", "ema_slow", "rsi_period"}:
                value = float(int(round(value)))
            decoded[name] = float(np.clip(value, low, high))
        return decoded

    @staticmethod
    def _clean_metric(value: Any, *, default: float = 0.0, cap: float | None = None) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive cleaning
            return default
        if math.isnan(numeric):
            return default
        if math.isinf(numeric):
            numeric = cap if cap is not None else default
        if cap is not None:
            numeric = max(-cap, min(cap, numeric))
        return numeric

    def _evaluate(self, params: Dict[str, float], *, method: str) -> Evaluation:
        key = self._make_key(params)
        cached = self._cache.get(key)
        if cached is not None:
            self._store_record(key, method, cached)
            return cached

        ema_fast = int(round(params["ema_fast"]))
        ema_slow = int(round(params["ema_slow"]))
        rsi_period = int(round(params["rsi_period"]))
        atr_mult_sl = float(params["atr_mult_sl"])
        atr_mult_tp = float(params["atr_mult_tp"])

        strategy_params = {
            "SMA Crossover": {"fast": float(ema_fast), "slow": float(ema_slow)},
            "RSI": {"period": float(rsi_period)},
        }

        try:
            result = run_backtest(
                self.data,
                self.strategies,
                strategy_params,
                capital_inicial=self.capital_inicial,
                riesgo_por_trade=self.riesgo_por_trade,
                sl_ratio=atr_mult_sl,
                tp_ratio=atr_mult_tp,
                logica=self.logica,
                symbol=self.symbol,
            )
            metrics = result.metrics
        except Exception:  # pragma: no cover - robustness for UI execution
            metrics = {}

        expectancy = self._clean_metric(metrics.get("Expectancy R"), default=0.0)
        drawdown = self._clean_metric(metrics.get("Max Drawdown %"), default=100.0)
        profit_factor = self._clean_metric(
            metrics.get("Profit Factor"), default=1.0, cap=10.0
        )
        rr_effective = self._clean_metric(
            metrics.get("RR Effective"), default=1.0, cap=10.0
        )

        fitness = expectancy - 0.1 * (drawdown / 10.0) + 0.05 * (profit_factor - 1.0)
        mark_as_best = expectancy >= 0.20 and drawdown <= 10.0
        meets_target = (
            expectancy >= 0.25
            and drawdown <= 10.0
            and profit_factor >= 1.4
            and rr_effective >= 1.5
        )

        evaluation = Evaluation(
            params={
                "ema_fast": float(ema_fast),
                "ema_slow": float(ema_slow),
                "rsi_period": float(rsi_period),
                "atr_mult_sl": atr_mult_sl,
                "atr_mult_tp": atr_mult_tp,
            },
            expectancy_R=expectancy,
            max_drawdown=drawdown,
            profit_factor=profit_factor,
            rr_effective=rr_effective,
            fitness=fitness,
            mark_as_best=mark_as_best,
            meets_target=meets_target,
        )
        self._cache[key] = evaluation
        self._store_record(key, method, evaluation)
        return evaluation

    # ------------------------------------------------------------------
    # Genetic algorithm implementation
    # ------------------------------------------------------------------
    def _run_genetic(
        self,
        *,
        population_size: int,
        generations: int,
        elite_ratio: float,
        random_state: int,
    ) -> None:
        if base is None or creator is None or tools is None:  # pragma: no cover
            raise ImportError(
                "La dependencia 'deap' es requerida para el modo genético."
            )

        random.seed(random_state)

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(PARAM_BOUNDS)
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate_individual(individual: Sequence[float]) -> Tuple[float]:
            params = self._decode_individual(individual)
            evaluation = self._evaluate(params, method="Genético")
            return (evaluation.fitness,)

        def mutate_individual(individual: Sequence[float]) -> Tuple[Sequence[float]]:
            for idx in range(len(individual)):
                individual[idx] += random.uniform(-0.1, 0.1)
                individual[idx] = min(1.0, max(0.0, individual[idx]))
            return (individual,)

        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=population_size)
        elite_count = max(1, int(population_size * elite_ratio))

        # Evaluate initial population
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]
        for individual, fitness in zip(
            invalid_individuals, map(toolbox.evaluate, invalid_individuals)
        ):
            individual.fitness.values = fitness

        for _ in range(generations):
            elites = tools.selBest(population, elite_count)
            offspring = toolbox.select(population, population_size - elite_count)
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_offspring = [ind for ind in offspring if not ind.fitness.valid]
            for individual, fitness in zip(
                invalid_offspring, map(toolbox.evaluate, invalid_offspring)
            ):
                individual.fitness.values = fitness

            population[:] = elites + offspring[: population_size - elite_count]

        tools.selBest(population, elite_count)  # ensure final elites evaluated

    # ------------------------------------------------------------------
    # Bayesian optimisation implementation
    # ------------------------------------------------------------------
    def _run_bayesian(self, *, n_calls: int, random_state: int) -> None:
        if gp_minimize is None or Integer is None or Real is None:  # pragma: no cover
            raise ImportError(
                "La dependencia 'scikit-optimize' es requerida para el modo bayesiano."
            )

        space = [
            Integer(10, 30, name="ema_fast"),
            Integer(50, 150, name="ema_slow"),
            Integer(7, 21, name="rsi_period"),
            Real(1.5, 3.0, name="atr_mult_sl"),
            Real(3.0, 6.0, name="atr_mult_tp"),
        ]

        def objective(values: List[float]) -> float:
            params = {name: float(value) for name, value in zip(PARAM_BOUNDS, values)}
            evaluation = self._evaluate(params, method="Bayesiano")
            return -evaluation.fitness

        gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=random_state,
            n_initial_points=min(10, n_calls),
        )


__all__ = ["AdvancedOptimizer", "PARAM_BOUNDS", "RESULTS_PATH"]
