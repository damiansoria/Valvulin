"""Funciones de backtesting visual para estrategias combinadas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    """Agrupa los resultados principales de un backtest visual."""

    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.Series
    data: pd.DataFrame
    drawdown: pd.Series


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Convierte ``open_time`` en un índice temporal legible para las gráficas."""

    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    converted = pd.to_datetime(series, errors="coerce")
    if converted.notna().any():
        return converted

    # Último intento asumiendo timestamps en milisegundos (formato Binance).
    converted_ms = pd.to_datetime(series, unit="ms", errors="coerce")
    return converted_ms


def _strategy_signal(df: pd.DataFrame, strategy: str, params: Dict[str, float]) -> pd.Series:
    """Calcula señales discretas (-1, 0, 1) para una estrategia concreta."""

    signal = pd.Series(0, index=df.index, dtype="float64")

    if strategy == "SMA Crossover":
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 50))
        df[f"sma_fast_{fast}"] = df["close"].rolling(fast).mean()
        df[f"sma_slow_{slow}"] = df["close"].rolling(slow).mean()
        fast_col = df[f"sma_fast_{fast}"]
        slow_col = df[f"sma_slow_{slow}"]
        signal.loc[fast_col > slow_col] = 1
        signal.loc[fast_col < slow_col] = -1

    elif strategy == "RSI":
        period = int(params.get("period", 14))
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        rsi_col = df[f"rsi_{period}"]
        signal.loc[rsi_col < 30] = 1
        signal.loc[rsi_col > 70] = -1

    elif strategy == "Bollinger Bands":
        period = int(params.get("period", 20))
        std_mult = float(params.get("std_mult", 2))
        ma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        df[f"bb_ma_{period}"] = ma
        df[f"bb_upper_{period}"] = ma + std_mult * std
        df[f"bb_lower_{period}"] = ma - std_mult * std
        signal.loc[df["close"] < df[f"bb_lower_{period}"]] = 1
        signal.loc[df["close"] > df[f"bb_upper_{period}"]] = -1

    return signal.fillna(0).astype(int)


def calculate_metrics(trades_df: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, float]:
    """Calcula métricas avanzadas inspiradas en TradingView."""

    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    metrics = {
        "Total Trades": total_trades,
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Winrate %": round((len(wins) / total_trades) * 100, 2) if total_trades else 0.0,
        "Profit Factor": round(
            wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else np.inf,
            2,
        ),
        "Total Return %": round((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100, 2)
        if len(equity_curve) > 1
        else 0.0,
        "Max Drawdown %": round((1 - equity_curve / equity_curve.cummax()).max() * 100, 2),
        "Average Win %": round(wins["pnl"].mean() * 100, 2) if len(wins) else 0.0,
        "Average Loss %": round(losses["pnl"].mean() * 100, 2) if len(losses) else 0.0,
        "Largest Win %": round(wins["pnl"].max() * 100, 2) if len(wins) else 0.0,
        "Largest Loss %": round(losses["pnl"].min() * 100, 2) if len(losses) else 0.0,
        "Expectancy %": round(trades_df["pnl"].mean() * 100, 2) if total_trades else 0.0,
        "Sharpe Ratio": round(
            (trades_df["pnl"].mean() / trades_df["pnl"].std()) * np.sqrt(252),
            2,
        )
        if trades_df["pnl"].std() > 0
        else 0.0,
        "Equity Final": round(float(equity_curve.iloc[-1]), 2),
    }
    return metrics


def _normalize_strategies(strategies: Sequence[str] | str) -> List[str]:
    """Normaliza la entrada de estrategias a una lista explícita."""

    if isinstance(strategies, str):
        return [strategies]
    return [item for item in strategies if item]


def run_backtest(
    data: pd.DataFrame,
    strategies: Sequence[str] | str,
    params: Dict[str, Dict[str, float]] | Dict[str, float] | None,
    capital_inicial: float = 1_000.0,
    riesgo_por_trade: float = 1.0,
    logica: str = "AND",
) -> StrategyResult:
    """Ejecuta un backtest incorporando simulación monetaria y métricas completas."""

    df = data.copy().reset_index(drop=True)

    if "open_time" in df.columns:
        df["open_time"] = _ensure_datetime(df["open_time"])

    df.sort_values(by="open_time", inplace=True, ignore_index=True)

    strategy_list = _normalize_strategies(strategies)
    if not strategy_list:
        raise ValueError("Debes seleccionar al menos una estrategia para el backtest.")

    params = params or {}
    signals_cols: List[str] = []

    for idx, strategy in enumerate(strategy_list):
        if isinstance(params.get(strategy), dict):
            strategy_params = params[strategy]  # type: ignore[index]
        else:
            strategy_params = params if isinstance(params, dict) else {}

        signal_col = f"signal_{idx}"
        df[signal_col] = _strategy_signal(df, strategy, strategy_params)
        signals_cols.append(signal_col)

    combined = pd.DataFrame({col: df[col] for col in signals_cols})
    long_mask = combined.eq(1)
    short_mask = combined.eq(-1)

    if logica.upper() == "AND":
        long_condition = long_mask.all(axis=1)
        short_condition = short_mask.all(axis=1)
    else:
        long_condition = long_mask.any(axis=1)
        short_condition = short_mask.any(axis=1)

    df["signal"] = 0
    df.loc[long_condition & ~short_condition, "signal"] = 1
    df.loc[short_condition & ~long_condition, "signal"] = -1

    capital = float(capital_inicial)
    equity_values: List[float] = [capital]
    trades: List[Dict[str, float | str | pd.Timestamp]] = []

    position = 0
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    entry_capital = capital

    riesgo_fraccion = float(riesgo_por_trade) / 100

    for _, row in df.iterrows():
        signal = int(row.get("signal", 0))
        price = float(row.get("close", 0.0))
        timestamp = row.get("open_time")

        if position == 0:
            if signal != 0:
                position = signal
                entry_price = price
                entry_time = timestamp
                entry_capital = capital
        else:
            if signal == position:
                continue

            trade_return = (price - entry_price) / entry_price * position
            pnl_pct = trade_return * riesgo_fraccion
            pnl_usd = capital * pnl_pct
            capital *= 1 + pnl_pct
            trades.append(
                {
                    "entrada": entry_time,
                    "salida": timestamp,
                    "lado": "Largo" if position == 1 else "Corto",
                    "precio_entrada": round(entry_price, 6),
                    "precio_salida": round(price, 6),
                    "retorno_activo_%": round(trade_return * 100, 4),
                    "pnl": pnl_pct,
                    "pnl_usd": round(pnl_usd, 4),
                    "capital_inicial": round(entry_capital, 4),
                    "capital_final": round(capital, 4),
                }
            )
            equity_values.append(capital)

            if signal == 0:
                position = 0
                entry_price = 0.0
                entry_time = None
            else:
                position = signal
                entry_price = price
                entry_time = timestamp
                entry_capital = capital

    if position != 0 and entry_price > 0:
        last_row = df.iloc[-1]
        price = float(last_row.get("close", entry_price))
        timestamp = last_row.get("open_time")
        trade_return = (price - entry_price) / entry_price * position
        pnl_pct = trade_return * riesgo_fraccion
        pnl_usd = capital * pnl_pct
        capital *= 1 + pnl_pct
        trades.append(
            {
                "entrada": entry_time,
                "salida": timestamp,
                "lado": "Largo" if position == 1 else "Corto",
                "precio_entrada": round(entry_price, 6),
                "precio_salida": round(price, 6),
                "retorno_activo_%": round(trade_return * 100, 4),
                "pnl": pnl_pct,
                "pnl_usd": round(pnl_usd, 4),
                "capital_inicial": round(entry_capital, 4),
                "capital_final": round(capital, 4),
            }
        )
        equity_values.append(capital)

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        trades_df["entrada"] = pd.to_datetime(trades_df["entrada"], errors="coerce")
        trades_df["salida"] = pd.to_datetime(trades_df["salida"], errors="coerce")

    equity_curve = pd.Series(equity_values, name="capital")
    drawdown = 1 - equity_curve / equity_curve.cummax()

    metrics = calculate_metrics(
        trades_df if not trades_df.empty else pd.DataFrame(columns=["pnl"]),
        equity_curve,
    )

    return StrategyResult(
        trades=trades_df,
        metrics=metrics,
        equity_curve=equity_curve,
        data=df,
        drawdown=drawdown,
    )
