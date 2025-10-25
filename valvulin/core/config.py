"""Configuration utilities for the Valvulin trading bot."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "config.yaml"
ENV_FILE = CONFIG_DIR / ".env"


@dataclass
class APISettings:
    """Holds API configuration values for exchanges and databases."""

    binance_api_key: str
    binance_api_secret: str
    database_url: str


@dataclass
class RiskSettings:
    """Stores the default risk parameters used by the risk engine."""

    max_position_size_pct: float
    max_daily_loss_pct: float
    leverage: float


@dataclass
class DataSettings:
    """Represents locations of data directories used by the bot."""

    cache_dir: Path
    export_dir: Path


@dataclass
class AppConfig:
    """Top-level configuration object."""

    api: APISettings
    risk: RiskSettings
    data: DataSettings
    backtester: Dict[str, Any]


class ConfigError(RuntimeError):
    """Raised when the configuration cannot be loaded or is invalid."""


class ConfigLoader:
    """Loads YAML configuration merged with environment variables."""

    def __init__(self, config_path: Optional[Path] = None, env_file: Optional[Path] = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_FILE
        self.env_file = env_file or ENV_FILE

    def load(self) -> AppConfig:
        """Load application configuration from YAML and environment variables."""

        config = self._load_yaml()
        env = self._load_env()
        api_settings = APISettings(
            binance_api_key=env.get("BINANCE_API_KEY", config["api"].get("binance_api_key", "")),
            binance_api_secret=env.get("BINANCE_API_SECRET", config["api"].get("binance_api_secret", "")),
            database_url=env.get("DATABASE_URL", config["api"].get("database_url", "sqlite:///valvulin.db")),
        )
        risk_cfg = config.get("risk", {})
        data_cfg = config.get("data", {})
        backtester_cfg = config.get("backtester", {})

        risk_settings = RiskSettings(
            max_position_size_pct=float(risk_cfg.get("max_position_size_pct", 2.0)),
            max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 3.0)),
            leverage=float(risk_cfg.get("leverage", 1.0)),
        )
        data_settings = DataSettings(
            cache_dir=Path(data_cfg.get("cache_dir", "data/cache")).expanduser().resolve(),
            export_dir=Path(data_cfg.get("export_dir", "data/exports")).expanduser().resolve(),
        )

        return AppConfig(
            api=api_settings,
            risk=risk_settings,
            data=data_settings,
            backtester=backtester_cfg,
        )

    def _load_yaml(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise ConfigError(f"Configuration file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _load_env(self) -> Dict[str, str]:
        env: Dict[str, str] = {}
        if self.env_file.exists():
            for line in self.env_file.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
        env.update({key: value for key, value in os.environ.items() if key.startswith("BINANCE_") or key.endswith("DATABASE_URL")})
        return env


__all__ = ["AppConfig", "ConfigLoader", "ConfigError", "APISettings", "RiskSettings", "DataSettings"]
