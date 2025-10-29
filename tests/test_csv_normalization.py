from __future__ import annotations

import pandas as pd

from analytics.csv_normalization import normalize_trade_dataframe


def test_normalize_trade_dataframe_handles_spanish_headers() -> None:
    df = pd.DataFrame(
        {
            "símbolo": ["BTCUSDT", "ETHUSDT"],
            "fecha_hora": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
            "lado": ["Largo", "Corto"],
            "precio_entrada": [100.0, 1500.0],
            "precio_salida": [110.0, 1400.0],
            "r_multiple": [1.0, -0.5],
            "capital_final": [1010.0, 955.0],
        }
    )

    normalised, applied = normalize_trade_dataframe(df)

    assert applied is True
    expected_columns = {
        "symbol",
        "timestamp",
        "side",
        "entry_price",
        "exit_price",
        "r_multiple",
        "capital_final",
    }
    assert expected_columns.issubset(set(normalised.columns))
    assert list(normalised["symbol"]) == ["BTCUSDT", "ETHUSDT"]


def test_normalize_trade_dataframe_is_idempotent_with_canonical_headers() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timestamp": ["2024-01-01 00:00:00"],
            "side": ["long"],
            "entry_price": [100.0],
            "exit_price": [110.0],
            "r_multiple": [1.0],
            "capital_final": [1010.0],
        }
    )

    normalised, applied = normalize_trade_dataframe(df)

    assert applied is False
    assert list(normalised.columns) == list(df.columns)
    pd.testing.assert_frame_equal(normalised, df)
