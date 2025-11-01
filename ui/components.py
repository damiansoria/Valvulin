"""Reusable Streamlit UI components for Valvulin Pro."""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .layout import card
from .theme import CARD_BG, TEXT_SECONDARY


def chart_block(
    title: str,
    caption: str,
    figure: go.Figure | None,
    *,
    use_container_width: bool = True,
    height: int | None = None,
) -> None:
    """Render a Plotly chart within a themed card."""

    with card(title, caption):
        if figure is None:
            st.info("⚠️ No hay datos disponibles para esta visualización.")
            return
        if height is not None:
            figure.update_layout(height=height)
        st.plotly_chart(figure, use_container_width=use_container_width)


def metric_block(title: str, description: str, metrics: Mapping[str, str | float | int]) -> None:
    """Render a metrics card with a description and Streamlit metrics."""

    with card(title, description):
        columns = st.columns(len(metrics)) if metrics else []
        for (label, value), column in zip(metrics.items(), columns):
            with column:
                st.metric(label=label, value=value)


def dataframe_block(df: pd.DataFrame, caption_text: str | None = None) -> None:
    """Render a dataframe with custom styling for positive/negative rows."""

    styled_df = _style_dataframe(df)
    if caption_text:
        st.caption(caption_text)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def _style_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    styler = df.style.set_table_attributes('class="valvulin-dataframe"').set_properties(
        **{"background-color": CARD_BG, "color": "#ECEFF4"}
    )

    highlight_columns = [
        col
        for col in df.columns
        if col.lower().startswith("pnl") or "resultado" in col.lower()
    ]

    def highlight_row(row: pd.Series) -> pd.Series:
        if highlight_columns:
            reference_col = highlight_columns[0]
            value = row.get(reference_col)
        else:
            numeric_values = pd.to_numeric(row, errors="coerce").dropna()
            value = numeric_values.iloc[0] if not numeric_values.empty else None

        is_positive = False
        if value is not None:
            try:
                is_positive = float(value) >= 0
            except (TypeError, ValueError):
                is_positive = False
        if value is None:
            joined = " ".join(row.astype(str).tolist())
            is_positive = "✅" in joined

        css_class = (
            "valvulin-highlight-positive" if is_positive else "valvulin-highlight-negative"
        )
        return pd.Series(css_class, index=row.index)

    try:
        class_matrix = df.apply(highlight_row, axis=1)
        styler = styler.set_td_classes(class_matrix)
    except Exception:
        # As a fallback we do not apply row highlighting
        styler = styler

    return styler


def trade_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a trade dataframe with emoji indicators for winners/losers."""

    if trades_df.empty:
        return trades_df

    df = trades_df.copy()
    if "pnl" in df.columns:
        df["Resultado"] = np.where(df["pnl"] >= 0, "✅ Ganadora", "❌ Perdedora")
        df["pnl_%"] = (df["pnl"] * 100).round(2)
    if "entrada" in df.columns and np.issubdtype(df["entrada"].dtype, np.datetime64):
        df["entrada"] = df["entrada"].dt.strftime("%Y-%m-%d %H:%M")
    if "salida" in df.columns and np.issubdtype(df["salida"].dtype, np.datetime64):
        df["salida"] = df["salida"].dt.strftime("%Y-%m-%d %H:%M")
    reorder = ["Resultado", *[col for col in df.columns if col not in {"Resultado"}]]
    return df[reorder]


def info_badge(text: str) -> None:
    st.markdown(
        f"<span style='background: rgba(255,255,255,0.05); padding: 0.4rem 0.8rem; border-radius: 999px; color:{TEXT_SECONDARY};'>"
        f"{text}</span>",
        unsafe_allow_html=True,
    )


__all__ = [
    "chart_block",
    "metric_block",
    "dataframe_block",
    "trade_table",
    "info_badge",
]
