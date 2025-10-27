"""Configuration loading utilities for the Valvulin trading engine.

The module is responsible for loading YAML/JSON configuration files,
applying CLI overrides, and exposing convenient dataclasses that can be
consumed by the rest of the system.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_LOGGER = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    yaml = None  # type: ignore


class ConfigError(RuntimeError):
    """Raised when configuration files cannot be parsed."""


@dataclass
class StrategyConfig:
    """Runtime information describing how a strategy should be instantiated."""

    name: str
    symbol: str
    timeframe: str
    handler: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.symbol}:{self.timeframe}:{self.name}"


@dataclass
class DataFeedConfig:
    name: str
    handler: str
    symbols: Sequence[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    type: Optional[str] = None


@dataclass
class SchedulerInterval:
    timeframe: str
    seconds: int
    warmup_seconds: Optional[int] = None


@dataclass
class SchedulerConfig:
    intervals: Sequence[SchedulerInterval] = field(default_factory=list)


@dataclass
class BotConfig:
    data_feeds: Sequence[DataFeedConfig]
    strategies: Sequence[StrategyConfig]
    risk: Dict[str, Any]
    execution: Dict[str, Any]
    scheduler: SchedulerConfig

    def find_strategy(self, symbol: str, timeframe: str, name: str) -> Optional[StrategyConfig]:
        key = f"{symbol}:{timeframe}:{name}"
        for strategy in self.strategies:
            if strategy.key() == key:
                return strategy
        return None

    def set_strategy_enabled(self, symbol: str, timeframe: str, name: str, enabled: bool) -> None:
        strategy = self.find_strategy(symbol, timeframe, name)
        if not strategy:
            raise ConfigError(
                f"Strategy '{name}' for {symbol} {timeframe} not found."
            )
        strategy.enabled = enabled


def _read_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file '{path}' does not exist")

    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ConfigError(
                    "PyYAML is required to parse YAML configuration files. Install it with 'pip install PyYAML'."
                )
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Unable to parse JSON configuration: {exc}") from exc
    except Exception as exc:
        if yaml is not None and isinstance(exc, yaml.YAMLError):  # pragma: no cover - defensive
            raise ConfigError(f"Unable to parse YAML configuration: {exc}") from exc
        raise

    raise ConfigError(
        f"Unsupported configuration format '{suffix}'. Use YAML or JSON."
    )


def _build_data_feeds(raw: Iterable[Dict[str, Any]]) -> List[DataFeedConfig]:
    feeds = []
    for entry in raw:
        parameters = entry.get("parameters") or entry.get("params") or {}
        feed_type = entry.get("type")
        feeds.append(
            DataFeedConfig(
                name=entry["name"],
                handler=entry.get("handler", ""),
                symbols=entry.get("symbols", []),
                parameters=parameters,
                type=feed_type,
            )
        )
    return feeds


def _build_strategies(raw: Iterable[Dict[str, Any]]) -> List[StrategyConfig]:
    strategies = []
    for entry in raw:
        strategies.append(
            StrategyConfig(
                name=entry["name"],
                symbol=entry["symbol"],
                timeframe=str(entry["timeframe"]),
                handler=entry["handler"],
                enabled=entry.get("enabled", True),
                parameters=entry.get("parameters", {}),
            )
        )
    return strategies


def _build_scheduler(raw: Dict[str, Any]) -> SchedulerConfig:
    intervals = []
    for entry in raw.get("intervals", []):
        intervals.append(
            SchedulerInterval(
                timeframe=str(entry["timeframe"]),
                seconds=int(entry["seconds"]),
                warmup_seconds=entry.get("warmup_seconds"),
            )
        )
    return SchedulerConfig(intervals=intervals)


def load_config(path: str) -> BotConfig:
    data = _read_file(Path(path))
    data_feeds = _build_data_feeds(data.get("data_feeds", []))
    strategies = _build_strategies(data.get("strategies", []))
    risk = data.get("risk", {})
    execution = data.get("execution", {})
    scheduler = _build_scheduler(data.get("scheduler", {}))
    return BotConfig(
        data_feeds=data_feeds,
        strategies=strategies,
        risk=risk,
        execution=execution,
        scheduler=scheduler,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valvulin trading engine", add_help=True
    )
    parser.add_argument(
        "--config",
        "-c",
        default=os.environ.get("VALVULIN_CONFIG", "config.yaml"),
        help="Ruta al archivo de configuración YAML/JSON",
    )
    parser.add_argument(
        "--enable-strategy",
        action="append",
        default=[],
        metavar="SYMBOL:TIMEFRAME:NAME",
        help="Activa una estrategia específica (se puede repetir).",
    )
    parser.add_argument(
        "--disable-strategy",
        action="append",
        default=[],
        metavar="SYMBOL:TIMEFRAME:NAME",
        help="Desactiva una estrategia específica (se puede repetir).",
    )
    return parser


def _parse_strategy_triplet(entry: str) -> Sequence[str]:
    parts = entry.split(":")
    if len(parts) != 3:
        raise ConfigError(
            "Strategy override must follow SYMBOL:TIMEFRAME:NAME format."
        )
    return parts


def apply_cli_overrides(config: BotConfig, args: argparse.Namespace) -> BotConfig:
    for entry in getattr(args, "enable_strategy", []) or []:
        symbol, timeframe, name = _parse_strategy_triplet(entry)
        _LOGGER.info(
            "CLI override: enabling strategy %s for %s %s", name, symbol, timeframe
        )
        config.set_strategy_enabled(symbol, timeframe, name, True)

    for entry in getattr(args, "disable_strategy", []) or []:
        symbol, timeframe, name = _parse_strategy_triplet(entry)
        _LOGGER.info(
            "CLI override: disabling strategy %s for %s %s", name, symbol, timeframe
        )
        config.set_strategy_enabled(symbol, timeframe, name, False)

    return config


def load_from_args(args: Optional[Sequence[str]] = None) -> BotConfig:
    parser = create_parser()
    namespace = parser.parse_args(args=args)
    config = load_config(namespace.config)
    config = apply_cli_overrides(config, namespace)
    return config
