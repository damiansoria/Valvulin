"""Registro de data feeds históricos para backtesting y analytics."""
from __future__ import annotations

from typing import Any, Dict, Type

from .binance_public import BinancePublicDataFeed
from .feeds import CSVDataFeed


_REGISTRY: Dict[str, Type[Any]] = {
    "csv": CSVDataFeed,
    "binance-public": BinancePublicDataFeed,
}


def register_feed(feed_type: str, feed_class: Type[Any]) -> None:
    """Registrar un nuevo tipo de feed histórico."""

    _REGISTRY[feed_type] = feed_class


def get_feed_class(feed_type: str) -> Type[Any]:
    """Obtener la clase asociada a un tipo de feed."""

    try:
        return _REGISTRY[feed_type]
    except KeyError as exc:  # pragma: no cover - validación defensiva
        raise ValueError(f"Tipo de feed desconocido: {feed_type}") from exc


def create_feed(feed_type: str, **params: Any) -> Any:
    """Instanciar un feed histórico a partir de su tipo registrado."""

    feed_class = get_feed_class(feed_type)
    return feed_class(**params)


__all__ = ["create_feed", "get_feed_class", "register_feed"]

