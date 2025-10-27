"""Funciones de backtesting visual para estrategias sencillas."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(data: pd.DataFrame, strategy: str, params: dict) -> dict:
    """Ejecuta un backtest simple basado en indicadores clásicos.

    Parameters
    ----------
    data:
        Datos OHLCV en formato DataFrame. Debe contener las columnas
        ``open_time``, ``open``, ``high``, ``low`` y ``close``.
    strategy:
        Nombre de la estrategia seleccionada.
    params:
        Diccionario con parámetros adicionales para la estrategia.

    Returns
    -------
    dict
        Diccionario con DataFrames de operaciones, métricas agregadas y
        la curva de equity acumulada.
    """

    df = data.copy().reset_index(drop=True)
    df["signal"] = 0

    if strategy == "SMA Crossover":
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 50))
        df["sma_fast"] = df["close"].rolling(fast).mean()
        df["sma_slow"] = df["close"].rolling(slow).mean()
        df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
        df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1

    elif strategy == "RSI":
        period = int(params.get("period", 14))
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df.loc[df["rsi"] < 30, "signal"] = 1
        df.loc[df["rsi"] > 70, "signal"] = -1

    elif strategy == "Bollinger Bands":
        period = int(params.get("period", 20))
        std_mult = float(params.get("std_mult", 2))
        df["ma"] = df["close"].rolling(period).mean()
        df["upper"] = df["ma"] + std_mult * df["close"].rolling(period).std()
        df["lower"] = df["ma"] - std_mult * df["close"].rolling(period).std()
        df.loc[df["close"] < df["lower"], "signal"] = 1
        df.loc[df["close"] > df["upper"], "signal"] = -1

    trades = []
    position = 0
    entry_price = 0.0

    for _, row in df.iterrows():
        signal = row["signal"]
        price = row["close"]

        if position == 0 and signal != 0:
            position = signal
            entry_price = price
        elif position != 0 and signal != position:
            pnl = (price - entry_price) / entry_price * position
            trades.append({"entry": entry_price, "exit": price, "pnl": pnl})
            position = signal
            entry_price = price

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        metrics = {
            "Trades": int(len(trades_df)),
            "Winrate %": float((trades_df["pnl"] > 0).mean() * 100),
            "Profit Factor": float(
                trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
                / abs(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum() or 1e-9)
            ),
            "Total Return %": float(trades_df["pnl"].sum() * 100),
        }
    else:
        metrics = {"Trades": 0, "Winrate %": 0.0, "Profit Factor": 0.0, "Total Return %": 0.0}

    equity_series = trades_df["pnl"].cumsum() if not trades_df.empty else pd.Series([0])
    equity_curve = pd.DataFrame({"equity": equity_series})

    return {"trades": trades_df, "metrics": metrics, "equity_curve": equity_curve}
