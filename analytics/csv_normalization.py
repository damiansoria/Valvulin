"""Utilities for harmonising CSV schemas across languages.

The analytics stack expects trade and equity CSV files to follow a specific
set of column names (``symbol``, ``timestamp``, ``entry_price`` ...).  However,
users often export data with Spanish headers or with custom capitalisation.
This module provides helpers that map the most common variants to the
canonical English names before performing any validation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping

import pandas as pd

__all__ = [
    "normalize_trade_dataframe",
    "normalize_equity_dataframe",
]


def _strip_accents(value: str) -> str:
    """Return *value* lowercased and without accented characters."""

    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()


def _standardise_label(label: str) -> str:
    """Convert *label* into a normalised identifier used for lookups."""

    simplified = _strip_accents(str(label).strip())
    simplified = simplified.replace(" ", "_").replace("-", "_")
    simplified = re.sub(r"[^a-z0-9_]+", "", simplified)
    return simplified


def _build_alias_lookup(mapping: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """Generate a dictionary mapping normalised aliases to canonical names."""

    lookup: dict[str, str] = {}
    for canonical, aliases in mapping.items():
        lookup[_standardise_label(canonical)] = canonical
        for alias in aliases:
            lookup[_standardise_label(alias)] = canonical
    return lookup


_TRADE_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "symbol": ("symbol", "simbolo", "símbolo", "ticker", "activo", "par"),
    "timestamp": (
        "timestamp",
        "fecha_hora",
        "fechahora",
        "fecha",
        "hora",
        "entrada",
        "entry_time",
        "entrytime",
        "entry_timestamp",
        "open_time",
        "trade_time",
    ),
    "exit_timestamp": (
        "exit_timestamp",
        "exit_time",
        "salida",
        "fecha_salida",
        "salida_fecha",
    ),
    "side": ("side", "lado", "direccion", "dirección", "sentido"),
    "entry_price": (
        "entry_price",
        "precio_entrada",
        "precioentrada",
        "entrada_precio",
    ),
    "exit_price": (
        "exit_price",
        "precio_salida",
        "preciosalida",
        "salida_precio",
    ),
    "r_multiple": ("r_multiple", "rmultiple", "multiplo_r", "r"),
    "capital_final": (
        "capital_final",
        "capitalfinal",
        "equity_final",
        "equityfinal",
        "capital_fin",
    ),
    "capital_inicial": (
        "capital_inicial",
        "capitalinicial",
        "entry_equity",
        "equity_inicial",
    ),
    "strategy": ("strategy", "estrategia"),
    "pnl": ("pnl", "retorno", "retorno_pct", "retorno_porcentaje"),
    "pnl_usd": ("pnl_usd", "pnlusd", "beneficio_usd"),
    "quantity": ("quantity", "size", "cantidad", "position_size"),
}

_EQUITY_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "fecha_hora", "fechahora", "fecha", "hora"),
    "equity": ("equity", "capital", "capital_final", "equidad"),
    "drawdown": ("drawdown", "caida", "retroceso", "dd"),
}

_TRADE_ALIAS_LOOKUP = _build_alias_lookup(_TRADE_COLUMN_ALIASES)
_EQUITY_ALIAS_LOOKUP = _build_alias_lookup(_EQUITY_COLUMN_ALIASES)


def _normalize_dataframe(
    df: pd.DataFrame, alias_lookup: Mapping[str, str]
) -> tuple[pd.DataFrame, bool]:
    """Return a copy of *df* with its columns mapped to canonical names.

    Parameters
    ----------
    df:
        Input dataframe loaded from an arbitrary CSV export.
    alias_lookup:
        Dictionary mapping normalised column aliases to canonical labels.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        A pair containing the transformed dataframe and a boolean flag
        indicating whether any renaming was required.
    """

    if df.empty:
        return df.copy(), False

    normalised = pd.DataFrame(index=df.index)
    translation_applied = False

    for column in df.columns:
        original_name = str(column)
        key = _standardise_label(original_name)
        target_name = alias_lookup.get(key, original_name)

        if target_name != original_name:
            translation_applied = True

        series = df[column]
        if target_name in normalised.columns:
            normalised[target_name] = normalised[target_name].combine_first(series)
        else:
            normalised[target_name] = series

    return normalised, translation_applied


def normalize_trade_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Normalise trade CSV columns to the canonical English schema."""

    return _normalize_dataframe(df, _TRADE_ALIAS_LOOKUP)


def normalize_equity_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Normalise equity curve CSV columns (timestamp, equity, drawdown)."""

    return _normalize_dataframe(df, _EQUITY_ALIAS_LOOKUP)
