"""Módulo de backtesting visual para ejecutar estrategias simples sobre datos OHLCV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Representa el resultado del backtest."""

    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    processed_data: pd.DataFrame


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el DataFrame de entrada para asegurar columnas y formato esperado."""

    if "close" not in data.columns or "open_time" not in data.columns:
        raise ValueError("El DataFrame debe contener las columnas 'open_time' y 'close'.")

    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    frame.dropna(subset=["timestamp"], inplace=True)
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.dropna(subset=["close"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _sma_crossover(frame: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    """Aplica la estrategia de cruce de medias móviles simples."""

    fast = int(params.get("sma_fast", 20))
    slow = int(params.get("sma_slow", 50))
    if fast <= 0 or slow <= 0:
        raise ValueError("Los parámetros de SMA deben ser mayores a cero.")
    if fast >= slow:
        # Evita medias idénticas que impidan los cruces.
        slow = fast + 1

    frame["sma_fast"] = frame["close"].rolling(fast, min_periods=fast).mean()
    frame["sma_slow"] = frame["close"].rolling(slow, min_periods=slow).mean()

    frame["signal"] = 0
    bullish_cross = (frame["sma_fast"] > frame["sma_slow"]) & (
        frame["sma_fast"].shift(1) <= frame["sma_slow"].shift(1)
    )
    bearish_cross = (frame["sma_fast"] < frame["sma_slow"]) & (
        frame["sma_fast"].shift(1) >= frame["sma_slow"].shift(1)
    )
    frame.loc[bullish_cross, "signal"] = 1
    frame.loc[bearish_cross, "signal"] = -1
    return frame


def _rsi(frame: pd.DataFrame, period: int) -> pd.Series:
    """Calcula el RSI clásico usando medias exponenciales."""

    delta = frame["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _rsi_strategy(frame: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    """Estrategia RSI clásica de sobrecompra / sobreventa."""

    period = int(params.get("rsi_period", 14))
    lower = float(params.get("rsi_lower", 30))
    upper = float(params.get("rsi_upper", 70))
    if period <= 0:
        raise ValueError("El periodo de RSI debe ser mayor a cero.")
    frame["rsi"] = _rsi(frame, period)
    frame["signal"] = 0
    frame.loc[(frame["rsi"] < lower) & (frame["rsi"].shift(1) >= lower), "signal"] = 1
    frame.loc[(frame["rsi"] > upper) & (frame["rsi"].shift(1) <= upper), "signal"] = -1
    return frame


def _bollinger_strategy(frame: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    """Estrategia basada en bandas de Bollinger."""

    window = int(params.get("bb_window", 20))
    std_mult = float(params.get("bb_std", 2.0))
    if window <= 1:
        raise ValueError("La ventana de Bollinger debe ser mayor a uno.")

    mid = frame["close"].rolling(window, min_periods=window).mean()
    std = frame["close"].rolling(window, min_periods=window).std(ddof=0)
    frame["bb_mid"] = mid
    frame["bb_upper"] = mid + std_mult * std
    frame["bb_lower"] = mid - std_mult * std
    frame["signal"] = 0
    frame.loc[
        (frame["close"] < frame["bb_lower"]) & (frame["close"].shift(1) >= frame["bb_lower"].shift(1)),
        "signal",
    ] = 1
    frame.loc[
        (frame["close"] > frame["bb_upper"]) & (frame["close"].shift(1) <= frame["bb_upper"].shift(1)),
        "signal",
    ] = -1
    return frame


def _generate_trades(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye la lista de operaciones y la curva de equity."""

    trades: List[Dict[str, object]] = []
    equity_curve: List[Dict[str, object]] = []

    initial_capital = 1.0
    equity = initial_capital
    position_qty = 0.0
    entry_price = 0.0
    entry_time = None

    for _, row in frame.iterrows():
        price = row["close"]
        timestamp = row["timestamp"]
        signal = row.get("signal", 0)

        if position_qty > 0 and price > 0:
            equity = position_qty * price

        if signal == 1 and position_qty == 0 and price > 0:
            position_qty = equity / price
            entry_price = price
            entry_time = timestamp
        elif signal == -1 and position_qty > 0 and price > 0:
            equity = position_qty * price
            profit_pct = (price - entry_price) / entry_price * 100
            trades.append(
                {
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": timestamp,
                    "exit_price": price,
                    "return_pct": profit_pct,
                    "bars": (timestamp - entry_time).total_seconds() if entry_time is not None else 0,
                }
            )
            position_qty = 0.0
            entry_price = 0.0
            entry_time = None

        equity_curve.append({"timestamp": timestamp, "equity": equity})

    # Cierre de posición al finalizar la serie si queda abierta
    if position_qty > 0:
        last_row = frame.iloc[-1]
        price = last_row["close"]
        timestamp = last_row["timestamp"]
        equity = position_qty * price if price > 0 else equity
        profit_pct = (price - entry_price) / entry_price * 100 if entry_price else 0.0
        trades.append(
            {
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": timestamp,
                "exit_price": price,
                "return_pct": profit_pct,
                "bars": (timestamp - entry_time).total_seconds() if entry_time is not None else 0,
            }
        )
        position_qty = 0.0
        equity_curve[-1]["equity"] = equity

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["bars"] = trades_df["bars"].apply(lambda seconds: seconds / 60 if seconds else 0)
        trades_df.rename(columns={"bars": "duracion_min"}, inplace=True)

    equity_df = pd.DataFrame(equity_curve)
    equity_df.drop_duplicates(subset="timestamp", inplace=True)
    equity_df.sort_values("timestamp", inplace=True)
    equity_df.reset_index(drop=True, inplace=True)
    return trades_df, equity_df


def _calculate_metrics(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> Dict[str, float]:
    """Calcula métricas básicas de rendimiento."""

    metrics: Dict[str, float] = {
        "total_return_pct": 0.0,
        "profit_factor": np.nan,
        "winrate": np.nan,
        "num_trades": float(len(trades)),
        "max_drawdown_pct": np.nan,
    }

    metrics["num_trades"] = int(metrics["num_trades"])

    if not equity_curve.empty:
        equity_series = equity_curve.set_index("timestamp")["equity"].astype(float)
        if not equity_series.empty:
            initial_equity = equity_series.iloc[0]
            final_equity = equity_series.iloc[-1]
            if initial_equity > 0:
                metrics["total_return_pct"] = (final_equity / initial_equity - 1) * 100
            running_max = equity_series.cummax()
            drawdown = (equity_series - running_max) / running_max
            if not drawdown.empty:
                metrics["max_drawdown_pct"] = drawdown.min() * 100

    if not trades.empty:
        profits = trades.loc[trades["return_pct"] > 0, "return_pct"].sum()
        losses = trades.loc[trades["return_pct"] < 0, "return_pct"].sum()
        losses_abs = abs(losses)
        if losses_abs > 0:
            metrics["profit_factor"] = profits / losses_abs
        elif trades["return_pct"].gt(0).all():
            metrics["profit_factor"] = float("inf")

        wins = (trades["return_pct"] > 0).sum()
        metrics["winrate"] = (wins / len(trades)) * 100

    return metrics


def run_backtest(data: pd.DataFrame, strategy: str, params: Dict[str, float]) -> Dict[str, object]:
    """Ejecuta el backtest usando un DataFrame con OHLCV y devuelve resultados listos para visualizar."""

    frame = _prepare_data(data)

    strategies = {
        "SMA Crossover": _sma_crossover,
        "RSI Oversold/Overbought": _rsi_strategy,
        "Bollinger Bands Reversal": _bollinger_strategy,
    }

    if strategy not in strategies:
        raise ValueError(f"Estrategia no soportada: {strategy}")

    frame = strategies[strategy](frame, params)
    trades_df, equity_df = _generate_trades(frame)
    metrics = _calculate_metrics(trades_df, equity_df)

    result = BacktestResult(
        trades=trades_df,
        metrics=metrics,
        equity_curve=equity_df,
        processed_data=frame,
    )

    return {
        "trades": result.trades,
        "metrics": result.metrics,
        "equity_curve": result.equity_curve,
        "processed_data": result.processed_data,
    }


__all__ = ["run_backtest"]

