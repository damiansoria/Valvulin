"""Interactive Plotly visualisations for backtest signals."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

PLOTS_DIR = Path(__file__).resolve().parent / "plots_output"


def _normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    rename_map = {col.lower(): col for col in frame.columns}
    required = {"open", "high", "low", "close"}
    if not required <= set(rename_map):
        missing = required - set(rename_map)
        raise ValueError(f"Missing OHLC columns: {', '.join(sorted(missing))}")
    return frame.rename(columns={rename_map[key]: key.capitalize() for key in required})


def _ensure_trades_frame(trades: Iterable | pd.DataFrame) -> pd.DataFrame:
    if isinstance(trades, pd.DataFrame):
        frame = trades.copy()
    else:
        rows = []
        for trade in trades:
            if hasattr(trade, "as_dict"):
                rows.append(trade.as_dict())
            elif is_dataclass(trade):
                rows.append(asdict(trade))
            elif isinstance(trade, Mapping):
                rows.append(dict(trade))
            else:
                raise TypeError(f"Unsupported trade type: {type(trade)!r}")
        frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def plot_backtest_signals(
    data: pd.DataFrame,
    trades: Sequence,
    indicators: Dict[str, pd.Series] | None = None,
    title: str = "Backtest Results",
) -> Path:
    """Render an interactive OHLC chart with trade signals and optional indicators.

    Parameters
    ----------
    data:
        DataFrame OHLC con las columnas `Open`, `High`, `Low`, `Close` y un índice
        compatible con fechas.
    trades:
        Lista o DataFrame con información de operaciones. Se espera disponer al
        menos de `entry_time`, `exit_time`, `entry_price`, `exit_price` y `side`.
    indicators:
        Diccionario opcional donde cada clave es el nombre del indicador y el
        valor es una serie de pandas alineada con `data`.
    title:
        Título del gráfico.

    Returns
    -------
    pathlib.Path
        Ruta del archivo HTML generado en ``analytics/plots_output``.
    """

    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("El DataFrame de datos debe estar indexado por fechas")

    frame = _normalise_columns(data)
    trades_df = _ensure_trades_frame(trades)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name="Precio",
        )
    )

    if indicators:
        for name, series in indicators.items():
            series = pd.Series(series)
            series = series.reindex(frame.index, method=None)
            fig.add_trace(
                go.Scatter(x=series.index, y=series.values, mode="lines", name=name)
            )

    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_time")
        side_series = trades_df.get("side")
        if side_series is None:
            side_series = pd.Series(["BUY"] * len(trades_df), index=trades_df.index)
        side_series = side_series.fillna("BUY").astype(str).str.upper()
        trades_df = trades_df.assign(_side=side_series)
        buys = trades_df[trades_df["_side"] == "BUY"]
        sells = trades_df[trades_df["_side"] == "SELL"]

        def _scatter(df: pd.DataFrame, *, is_entry: bool, side: str) -> go.Scatter:
            marker_symbol = "triangle-up" if side == "BUY" else "triangle-down"
            color = "#2b8a3e" if side == "BUY" else "#d9480f"
            column = "entry_time" if is_entry else "exit_time"
            price_col = "entry_price" if is_entry else "exit_price"
            name = f"{side.title()} {'entrada' if is_entry else 'salida'}"
            return go.Scatter(
                x=df[column],
                y=df[price_col],
                mode="markers",
                marker=dict(symbol=marker_symbol, color=color, size=11),
                name=name,
            )

        if not buys.empty:
            fig.add_trace(_scatter(buys, is_entry=True, side="BUY"))
            if {"exit_time", "exit_price"} <= set(buys.columns):
                fig.add_trace(_scatter(buys, is_entry=False, side="BUY"))
        if not sells.empty:
            fig.add_trace(_scatter(sells, is_entry=True, side="SELL"))
            if {"exit_time", "exit_price"} <= set(sells.columns):
                fig.add_trace(_scatter(sells, is_entry=False, side="SELL"))

        for _, trade in trades_df.iterrows():
            color = "#2b8a3e" if float(trade.get("pnl", 0.0)) >= 0 else "#d9480f"
            fig.add_trace(
                go.Scatter(
                    x=[
                        trade["entry_time"],
                        trade.get("exit_time", trade["entry_time"]),
                    ],
                    y=[
                        trade.get("entry_price"),
                        trade.get("exit_price", trade.get("entry_price")),
                    ],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    name="Trade",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = title.lower().replace(" ", "_").replace("/", "-")
    output_path = PLOTS_DIR / f"{slug}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


__all__ = ["plot_backtest_signals"]
