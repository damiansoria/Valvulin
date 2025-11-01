"""Utilities for applying the Valvulin Pro dashboard theme."""
from __future__ import annotations

from functools import lru_cache
from typing import Final

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PRIMARY_BG: Final[str] = "#0E1117"
CARD_BG: Final[str] = "#161B22"
TEXT_PRIMARY: Final[str] = "#F5F5F7"
TEXT_SECONDARY: Final[str] = "#B0B3B8"
ACCENT: Final[str] = "#FFD700"
EQUITY_COLOR: Final[str] = "#3A9FF5"
DRAWDOWN_COLOR: Final[str] = "#E74C3C"
PLOTLY_TEMPLATE_NAME: Final[str] = "valvulin_dark"


@lru_cache(maxsize=1)
def _register_plotly_template() -> None:
    """Register a reusable Plotly template with the Valvulin color palette."""

    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
        title=dict(font=dict(color=TEXT_PRIMARY, size=20)),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY),
        ),
        colorway=[EQUITY_COLOR, DRAWDOWN_COLOR, ACCENT, "#2ECC71", "#9B59B6"],
        margin=dict(l=40, r=40, t=60, b=40),
    )
    template.data.scatter = [
        go.Scatter(
            line=dict(width=2),
            marker=dict(size=8, opacity=0.85),
        )
    ]
    template.data.bar = [
        go.Bar(
            marker=dict(line=dict(width=0)),
        )
    ]
    template.data.histogram = [
        go.Histogram(marker=dict(line=dict(width=0))),
    ]

    pio.templates[PLOTLY_TEMPLATE_NAME] = template


def apply_theme() -> str:
    """Apply the dashboard theme and return the Plotly template name."""

    _register_plotly_template()

    st.markdown(
        f"""
        <style>
        :root {{
            --valvulin-bg: {PRIMARY_BG};
            --valvulin-card: {CARD_BG};
            --valvulin-text: {TEXT_PRIMARY};
            --valvulin-subtext: {TEXT_SECONDARY};
            --valvulin-accent: {ACCENT};
        }}
        .stApp {{
            background-color: var(--valvulin-bg);
            color: var(--valvulin-text);
        }}
        .stMarkdown, .stText, .stCaption, .stHtmlContent {{
            color: var(--valvulin-text) !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: var(--valvulin-text) !important;
        }}
        div[data-testid="metric-container"] {{
            background-color: var(--valvulin-card);
            padding: 1rem 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35);
        }}
        div[data-testid="metric-container"] label {{
            color: var(--valvulin-subtext) !important;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05rem;
        }}
        div[data-testid="metric-container"] .stMetric-value {{
            color: var(--valvulin-text) !important;
            font-size: 1.8rem;
            font-weight: 700;
        }}
        div[data-testid="metric-container"] .stMetricDelta {{
            color: var(--valvulin-accent) !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #11141B;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .stTabs [data-baseweb="tab"] button {{
            background: transparent;
            border-bottom: 3px solid transparent;
        }}
        .stTabs [data-baseweb="tab"] button:hover {{
            border-bottom: 3px solid rgba(255, 255, 255, 0.1);
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom: 3px solid var(--valvulin-accent) !important;
            color: var(--valvulin-accent) !important;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1600px;
        }}
        .valvulin-card {{
            background: var(--valvulin-card);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }}
        .valvulin-caption {{
            color: var(--valvulin-subtext) !important;
            font-size: 0.95rem;
        }}
        .valvulin-divider {{
            height: 1px;
            border: none;
            margin: 2rem 0;
            background: linear-gradient(90deg, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0));
        }}
        .valvulin-dataframe tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .valvulin-highlight-positive {{
            background-color: rgba(46, 204, 113, 0.15) !important;
            color: #D1F4E0 !important;
        }}
        .valvulin-highlight-negative {{
            background-color: rgba(231, 76, 60, 0.15) !important;
            color: #F5D0CD !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    return PLOTLY_TEMPLATE_NAME


__all__ = [
    "apply_theme",
    "PLOTLY_TEMPLATE_NAME",
    "ACCENT",
    "CARD_BG",
    "EQUITY_COLOR",
    "DRAWDOWN_COLOR",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
]
