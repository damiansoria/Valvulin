"""Funciones de backtesting visual para estrategias combinadas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.atr_dynamic import calculate_levels, compute_atr, position_size
from core.filters.trend_volatility_filter import allow_entry, build_context
from database import models as db_models


@dataclass
class StrategyResult:
    """Agrupa los resultados principales de un backtest visual."""

    trades: pd.DataFrame
    metrics: Dict[str, float | int | None]
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
    wins = trades_df[trades_df["pnl_usd"] > 0]
    losses = trades_df[trades_df["pnl_usd"] < 0]

    total_return_pct = (
        (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        if len(equity_curve) > 1
        else 0.0
    )
    total_return_usd = float(equity_curve.iloc[-1] - equity_curve.iloc[0])

    equity_cummax = equity_curve.cummax()
    drawdown_curve = (equity_curve - equity_cummax).fillna(0.0)
    drawdown_pct = (drawdown_curve / equity_cummax.replace(0, np.nan)).fillna(0.0)
    max_drawdown_usd = (
        abs(float(drawdown_curve.min())) if not drawdown_curve.empty else 0.0
    )
    max_drawdown_pct = (
        abs(float(drawdown_pct.min()) * 100) if not drawdown_pct.empty else 0.0
    )

    r_values = trades_df["r_multiple"].dropna()
    avg_r = float(r_values.mean()) if not r_values.empty else 0.0
    avg_win_r = (
        float(r_values[r_values > 0].mean()) if (r_values > 0).any() else 0.0
    )
    avg_loss_r = (
        float(r_values[r_values < 0].mean()) if (r_values < 0).any() else 0.0
    )
    winrate = (len(wins) / total_trades * 100) if total_trades else 0.0
    lossrate = 100 - winrate if total_trades else 0.0
    expectancy_r = (
        (winrate / 100) * avg_win_r + (lossrate / 100) * avg_loss_r
        if total_trades
        else 0.0
    )
    rr_effective = avg_win_r / abs(avg_loss_r) if avg_loss_r != 0 else np.inf
    breakeven_winrate = (
        abs(avg_loss_r) / (abs(avg_loss_r) + avg_win_r) * 100
        if (avg_win_r + abs(avg_loss_r)) > 0
        else 0.0
    )
    expectancy_usd = float(trades_df["pnl_usd"].mean()) if total_trades else 0.0

    gross_profit = wins["pnl_usd"].sum()
    gross_loss = abs(losses["pnl_usd"].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf

    avg_trade_pct = float(trades_df["pnl"].mean() * 100) if total_trades else 0.0
    median_trade_pct = float(trades_df["pnl"].median() * 100) if total_trades else 0.0
    sharpe_ratio = 0.0
    if total_trades > 1:
        pnl_std = trades_df["pnl"].std(ddof=0)
        if pnl_std > 0:
            sharpe_ratio = float((trades_df["pnl"].mean() / pnl_std) * np.sqrt(252))

    cumulative_expectancy = float(r_values.sum()) if not r_values.empty else 0.0

    monthly_expectancies: List[float] = []
    monthly_profit_factors: List[float] = []
    monthly_returns: Dict[str, float] = {}
    if not trades_df.empty and "timestamp" in trades_df:
        monthly = trades_df.copy()
        monthly["timestamp"] = pd.to_datetime(monthly["timestamp"], errors="coerce")
        monthly["month"] = monthly["timestamp"].dt.to_period("M")
        for month, group in monthly.groupby("month"):
            month_key = str(month)
            r_group = group["r_multiple"].dropna()
            if not r_group.empty:
                monthly_expectancies.append(float(r_group.mean()))
            gross_profit_m = group[group["pnl_usd"] > 0]["pnl_usd"].sum()
            gross_loss_m = abs(group[group["pnl_usd"] < 0]["pnl_usd"].sum())
            if gross_loss_m > 0:
                monthly_profit_factors.append(float(gross_profit_m / gross_loss_m))
            elif gross_profit_m > 0:
                monthly_profit_factors.append(float("inf"))
            monthly_returns[month_key] = float(group["pnl_usd"].sum())

    metrics = {
        "Total Trades": total_trades,
        "Winning Trades": int(len(wins)),
        "Losing Trades": int(len(losses)),
        "Winrate %": round(winrate, 2),
        "Profit Factor": round(profit_factor, 2)
        if np.isfinite(profit_factor)
        else np.inf,
        "Average R Multiple": round(avg_r, 3),
        "Expectancy R": round(expectancy_r, 3),
        "Expectancy $": round(expectancy_usd, 2),
        "Average Win (R)": round(avg_win_r, 3),
        "Average Loss (R)": round(avg_loss_r, 3),
        "RR Effective": round(rr_effective, 3) if np.isfinite(rr_effective) else np.inf,
        "Breakeven Winrate %": round(breakeven_winrate, 2),
        "Average Trade %": round(avg_trade_pct, 3),
        "Median Trade %": round(median_trade_pct, 3),
        "Max Drawdown %": round(max_drawdown_pct, 2),
        "Max Drawdown $": round(max_drawdown_usd, 2),
        "Total Return %": round(total_return_pct, 2),
        "Total Return $": round(total_return_usd, 2),
        "Equity Final $": round(float(equity_curve.iloc[-1]), 2),
        "Equity Final %": round(total_return_pct, 2),
        "Average Risk $": round(float(trades_df["risk_usd"].mean()), 2)
        if "risk_usd" in trades_df
        and not trades_df["risk_usd"].empty
        else 0.0,
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Expectancy R Cumulative": round(cumulative_expectancy, 3),
        "Monthly Expectancy R (avg)": round(float(np.mean(monthly_expectancies)), 3)
        if monthly_expectancies
        else 0.0,
        "Monthly Profit Factor (avg)": round(float(np.mean(monthly_profit_factors)), 3)
        if monthly_profit_factors
        else 0.0,
    }

    metrics.setdefault("Rolling Sharpe 30d", 0.0)

    if monthly_returns:
        best_month = max(monthly_returns.items(), key=lambda item: item[1])
        worst_month = min(monthly_returns.items(), key=lambda item: item[1])
        metrics["Best Month"] = f"{best_month[0]} ({best_month[1]:.2f} USD)"
        metrics["Worst Month"] = f"{worst_month[0]} ({worst_month[1]:.2f} USD)"
    else:
        metrics["Best Month"] = "-"
        metrics["Worst Month"] = "-"

    equity_returns = equity_curve.pct_change().dropna()
    if not equity_returns.empty:
        rolling_sharpe = equity_returns.rolling(window=30, min_periods=10).apply(
            lambda window: (window.mean() / window.std()) * np.sqrt(252)
            if window.std(ddof=0) > 0
            else 0.0,
            raw=False,
        )
        rolling_sharpe = rolling_sharpe.dropna()
        if not rolling_sharpe.empty:
            metrics["Rolling Sharpe 30d"] = round(float(rolling_sharpe.iloc[-1]), 3)

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
    sl_ratio: float = 1.0,
    tp_ratio: float = 2.0,
    logica: str = "AND",
    *,
    symbol: str | None = None,
) -> StrategyResult:
    """Ejecuta un backtest incorporando simulación monetaria y métricas completas."""

    df = data.copy().reset_index(drop=True)

    if "open_time" in df.columns:
        df["open_time"] = _ensure_datetime(df["open_time"])

    df.sort_values(by="open_time", inplace=True, ignore_index=True)

    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]

    strategy_list = _normalize_strategies(strategies)
    if not strategy_list:
        raise ValueError("Debes seleccionar al menos una estrategia para el backtest.")

    params = params or {}
    signals_cols: List[str] = []
    strategy_signal_map: Dict[str, str] = {}

    for idx, strategy in enumerate(strategy_list):
        if isinstance(params.get(strategy), dict):
            strategy_params = params[strategy]  # type: ignore[index]
        else:
            strategy_params = params if isinstance(params, dict) else {}

        signal_col = f"signal_{idx}"
        df[signal_col] = _strategy_signal(df, strategy, strategy_params)
        signals_cols.append(signal_col)
        strategy_signal_map[strategy] = signal_col

    atr_period = 14
    atr_series = compute_atr(df, period=atr_period)
    atr_column = atr_series.name
    df[atr_column] = atr_series

    filter_context = build_context(
        data=df,
        atr_period=atr_period,
        strategy_columns=strategy_signal_map,
    )
    use_full_atr = len(df) >= atr_period * 2

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
    symbol_value = symbol or ""
    trades: List[Dict[str, float | str | pd.Timestamp | None]] = []

    sl_ratio = float(sl_ratio) if sl_ratio > 0 else 1.0
    tp_ratio = float(tp_ratio) if tp_ratio > 0 else 2.0
    fallback_fraction = 0.01

    equity_values: List[float] = []
    equity_timestamps: List[pd.Timestamp] = []

    def _normalise_timestamp(value: object, fallback: Optional[pd.Timestamp]) -> pd.Timestamp:
        if isinstance(value, pd.Timestamp) and not pd.isna(value):
            return value
        if value is not None:
            converted = pd.to_datetime(value, errors="coerce")
            if not pd.isna(converted):
                return converted
        return fallback or pd.Timestamp.utcnow()

    initial_timestamp: Optional[pd.Timestamp] = None
    if "open_time" in df.columns and df["open_time"].notna().any():
        first_valid = df["open_time"].first_valid_index()
        if first_valid is not None:
            initial_timestamp = _normalise_timestamp(
                df.loc[first_valid, "open_time"], None
            )
    if initial_timestamp is None:
        initial_timestamp = pd.Timestamp.utcnow()

    equity_values.append(capital)
    equity_timestamps.append(initial_timestamp)

    position = 0
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None
    entry_capital = capital
    entry_risk_amount = 0.0
    entry_position_size = 0.0
    entry_stop_price: Optional[float] = None
    entry_take_price: Optional[float] = None
    entry_stop_distance: float = 0.0
    entry_atr_value: float = 0.0

    def _open_position(
        signal_value: int,
        price_value: float,
        timestamp_value: pd.Timestamp,
        atr_value: float,
    ) -> None:
        nonlocal position
        nonlocal entry_price
        nonlocal entry_time
        nonlocal entry_capital
        nonlocal entry_risk_amount
        nonlocal entry_position_size
        nonlocal entry_stop_price
        nonlocal entry_take_price
        nonlocal entry_stop_distance
        nonlocal entry_atr_value

        if signal_value == 0:
            return
        current_capital = max(capital, 0.0)
        if current_capital <= 0 or price_value <= 0:
            return
        atr_input = atr_value
        if (not use_full_atr) or atr_value <= price_value * 0.001:
            atr_input = price_value * fallback_fraction

        levels = calculate_levels(
            price=price_value,
            atr_value=atr_input,
            atr_mult_sl=sl_ratio,
            atr_mult_tp=tp_ratio,
            direction=signal_value,
        )
        if levels.stop_distance <= 0:
            return
        position_size_value = position_size(
            capital=current_capital,
            risk_percent=riesgo_por_trade,
            atr_value=atr_input,
            atr_mult_sl=sl_ratio,
        )
        if position_size_value <= 0:
            return
        position = signal_value
        entry_price = price_value
        entry_time = timestamp_value
        entry_capital = capital
        entry_risk_amount = position_size_value * levels.stop_distance
        entry_position_size = position_size_value
        entry_stop_price = levels.stop_loss
        entry_take_price = levels.take_profit
        entry_stop_distance = levels.stop_distance
        entry_atr_value = atr_input

    for idx, row in df.iterrows():
        signal = int(row.get("signal", 0))
        price = float(row.get("close", 0.0))
        timestamp = _normalise_timestamp(row.get("open_time"), equity_timestamps[-1])
        atr_value = float(row.get(atr_column, 0.0))

        if position == 0:
            if allow_entry(context=filter_context, index=idx, signal=signal, row=row):
                _open_position(signal, price, timestamp, atr_value)
            continue

        if signal == position:
            continue

        direction = 1 if position == 1 else -1
        current_position_size = entry_position_size
        if current_position_size <= 0:
            position = 0
            continue

        trade_return = (price - entry_price) / entry_price * direction
        pnl_usd = direction * (price - entry_price) * current_position_size
        stop_distance = entry_stop_distance if entry_stop_distance > 0 else abs(entry_price - (entry_stop_price or entry_price))
        risk_amount = (
            current_position_size * stop_distance if stop_distance > 0 else entry_risk_amount
        )
        r_multiple = (
            direction * (price - entry_price) / stop_distance
            if stop_distance > 0
            else np.nan
        )
        capital += pnl_usd
        pnl_pct = pnl_usd / entry_capital if entry_capital else 0.0
        trades.append(
            {
                "timestamp": entry_time,
                "exit_timestamp": timestamp,
                "entrada": entry_time,
                "salida": timestamp,
                "lado": "Largo" if direction == 1 else "Corto",
                "precio_entrada": entry_price,
                "precio_salida": price,
                "retorno_activo_%": trade_return * 100,
                "pnl": pnl_pct,
                "pnl_usd": pnl_usd,
                "r_multiple": r_multiple,
                "risk_usd": risk_amount,
                "position_size": current_position_size,
                "capital_inicial": entry_capital,
                "capital_final": capital,
                "stop_price": entry_stop_price,
                "take_profit_price": entry_take_price,
                "stop_distance": stop_distance,
                "risk_pct": riesgo_por_trade,
                "symbol": symbol_value,
                "strategy": "+".join(strategy_list),
                "atr_value": entry_atr_value,
            }
        )
        equity_values.append(capital)
        equity_timestamps.append(timestamp)

        if signal == 0:
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_risk_amount = 0.0
            entry_position_size = 0.0
            entry_stop_price = None
            entry_take_price = None
            entry_stop_distance = 0.0
            entry_atr_value = 0.0
        else:
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_risk_amount = 0.0
            entry_position_size = 0.0
            entry_stop_price = None
            entry_take_price = None
            entry_stop_distance = 0.0
            entry_atr_value = 0.0
            if allow_entry(context=filter_context, index=idx, signal=signal, row=row):
                _open_position(signal, price, timestamp, atr_value)

    if position != 0 and entry_price > 0:
        last_row = df.iloc[-1]
        price = float(last_row.get("close", entry_price))
        timestamp = _normalise_timestamp(last_row.get("open_time"), equity_timestamps[-1])
        direction = 1 if position == 1 else -1
        current_position_size = entry_position_size
        trade_return = (price - entry_price) / entry_price * direction
        pnl_usd = direction * (price - entry_price) * current_position_size
        stop_distance = entry_stop_distance if entry_stop_distance > 0 else abs(entry_price - (entry_stop_price or entry_price))
        risk_amount = (
            current_position_size * stop_distance if stop_distance > 0 else entry_risk_amount
        )
        r_multiple = (
            direction * (price - entry_price) / stop_distance
            if stop_distance > 0
            else np.nan
        )
        capital += pnl_usd
        pnl_pct = pnl_usd / entry_capital if entry_capital else 0.0
        trades.append(
            {
                "timestamp": entry_time,
                "exit_timestamp": timestamp,
                "entrada": entry_time,
                "salida": timestamp,
                "lado": "Largo" if direction == 1 else "Corto",
                "precio_entrada": entry_price,
                "precio_salida": price,
                "retorno_activo_%": trade_return * 100,
                "pnl": pnl_pct,
                "pnl_usd": pnl_usd,
                "r_multiple": r_multiple,
                "risk_usd": risk_amount,
                "position_size": current_position_size,
                "capital_inicial": entry_capital,
                "capital_final": capital,
                "stop_price": entry_stop_price,
                "take_profit_price": entry_take_price,
                "stop_distance": stop_distance,
                "risk_pct": riesgo_por_trade,
                "symbol": symbol_value,
                "strategy": "+".join(strategy_list),
                "atr_value": entry_atr_value,
            }
        )
        equity_values.append(capital)
        equity_timestamps.append(timestamp)

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        capital = float(capital_inicial)
        capital_series: List[float] = []
        capital_inicial_series: List[float] = []
        capital_final_series: List[float] = []

        for pnl in trades_df["pnl_usd"].astype(float):
            capital_inicial_series.append(capital)
            capital += pnl
            capital_final_series.append(capital)
            capital_series.append(capital)

        trades_df["capital_inicial"] = capital_inicial_series
        trades_df["capital_final"] = capital_final_series
        equity_values = [float(capital_inicial), *capital_series]

        trades_df["entrada"] = pd.to_datetime(trades_df["entrada"], errors="coerce")
        trades_df["salida"] = pd.to_datetime(trades_df["salida"], errors="coerce")

    if len(equity_timestamps) == len(equity_values):
        equity_index = pd.to_datetime(equity_timestamps)
        equity_curve = pd.Series(equity_values, index=equity_index, name="capital")
    else:
        equity_curve = pd.Series(equity_values, name="capital")

    equity_cummax = equity_curve.cummax().replace(0, np.nan)
    drawdown = ((equity_curve - equity_cummax) / equity_cummax).fillna(0.0)

    metrics = calculate_metrics(
        trades_df
        if not trades_df.empty
        else pd.DataFrame(columns=["pnl", "pnl_usd", "r_multiple", "risk_usd"]),
        equity_curve,
    )

    try:
        db_models.record_backtest_run(
            symbol=symbol_value or symbol,
            logica=logica,
            capital=capital_inicial,
            metrics=metrics,
        )
    except Exception:  # pragma: no cover - persistence errors should not break backtests
        pass

    return StrategyResult(
        trades=trades_df,
        metrics=metrics,
        equity_curve=equity_curve,
        data=df,
        drawdown=drawdown,
    )
