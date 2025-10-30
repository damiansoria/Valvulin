"""Reusable plotting utilities for the Streamlit analytics dashboard."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _resolve_x_values(df_trades: pd.DataFrame) -> Iterable:
    """Return the x-axis sequence for a trades dataframe."""

    if "timestamp" in df_trades.columns and df_trades["timestamp"].notna().any():
        return df_trades["timestamp"]
    return list(range(len(df_trades)))


def plot_equity_curve(
    df_trades: pd.DataFrame, *, drawdown: Optional[pd.Series] = None
) -> go.Figure:
    """Create an interactive equity curve highlighting trade outcomes."""

    x_values = _resolve_x_values(df_trades)
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=df_trades["capital_final"],
            mode="lines",
            line=dict(color="#1f77b4", width=3),
            name="Equity",
            hovertemplate="Fecha: %{x}<br>Capital: $%{y:,.2f}<extra></extra>",
        )
    )

    if not df_trades.empty and "r_multiple" in df_trades.columns:
        winners = df_trades[df_trades["r_multiple"] >= 0]
        losers = df_trades[df_trades["r_multiple"] < 0]
        for subset, name, color in (
            (winners, "Ganadora", "#2ecc71"),
            (losers, "Perdedora", "#e74c3c"),
        ):
            if subset.empty:
                continue
            pnl_pct = subset.get("pnl", pd.Series(np.nan, index=subset.index)) * 100
            r_mult = subset.get("r_multiple", pd.Series(np.nan, index=subset.index))
            custom = np.column_stack(
                [pnl_pct.fillna(0.0).astype(float), r_mult.fillna(0.0).astype(float)]
            )
            figure.add_trace(
                go.Scatter(
                    x=subset["timestamp"] if "timestamp" in subset.columns else subset.index,
                    y=subset["capital_final"],
                    mode="markers",
                    marker=dict(size=10, color=color, symbol="circle"),
                    name=f"{name} ({len(subset)})",
                    customdata=custom,
                    hovertemplate=(
                        "Fecha: %{x}<br>Capital: $%{y:,.2f}<br>Retorno: %{customdata[0]:.2f}%"
                        "<br>R múltiple: %{customdata[1]:.2f}<extra></extra>"
                    ),
                )
            )

    if drawdown is not None and not drawdown.empty:
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=drawdown,
                mode="lines",
                line=dict(color="#d62728", dash="dash"),
                name="Drawdown %",
                yaxis="y2",
                hovertemplate="Fecha: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
            )
        )

        figure.update_layout(
            yaxis=dict(title="Capital ($)"),
            yaxis2=dict(title="Drawdown (%)", overlaying="y", side="right", showgrid=False),
        )
    else:
        figure.update_layout(yaxis=dict(title="Capital ($)"))

    figure.update_layout(
        title="Equity Curve",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return figure


def plot_drawdown_curve(df_trades: pd.DataFrame) -> go.Figure:
    """Plot drawdown percentage over time with interactive hover."""

    x_values = _resolve_x_values(df_trades)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=df_trades["drawdown"],
            fill="tozeroy",
            line=dict(color="#ff7f0e"),
            name="Drawdown %",
            hovertemplate="Fecha: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title="Drawdown Curve",
        xaxis_title="Time" if "timestamp" in df_trades.columns else "Trade #",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return figure


def plot_r_multiple_distribution(
    df_trades: pd.DataFrame, *, bins: int = 30
) -> go.Figure:
    """Plot the distribution of R-multiples as a histogram."""

    r_values = df_trades["r_multiple"].dropna()
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=r_values,
            nbinsx=bins,
            marker=dict(color="#6ab7ff"),
            name="R",
            hovertemplate="R: %{x:.2f}<br>Frecuencia: %{y}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line=dict(color="#d62728", dash="dash"), annotation_text="Break-even")
    if not r_values.empty:
        figure.add_vline(
            x=float(r_values.mean()),
            line=dict(color="#2ecc71", dash="dot"),
            annotation_text="Media",
        )
    figure.update_layout(
        title="R-Multiple Distribution",
        xaxis_title="R-Multiple",
        yaxis_title="Frecuencia",
        bargap=0.1,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return figure


def plot_expectancy_bar(
    expectancy_value: float, *, ylim: tuple[float, float] = (-2.0, 2.0)
) -> go.Figure:
    """Return a bar chart visualising expectancy."""

    color = "#2ecc71" if expectancy_value >= 0 else "#e74c3c"
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=["Expectancy"],
            y=[expectancy_value],
            marker_color=color,
            hovertemplate="Expectancy: %{y:.2f}R<extra></extra>",
        )
    )
    figure.add_hline(y=0, line=dict(color="#333333", width=1))
    figure.update_layout(
        yaxis=dict(range=list(ylim), title="Expectancy (R)"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return figure


__all__ = [
    "plot_equity_curve",
    "plot_drawdown_curve",
    "plot_r_multiple_distribution",
    "plot_expectancy_bar",
]
