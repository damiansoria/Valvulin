"""Layout helpers for structuring the Valvulin Pro dashboard."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Tuple

import streamlit as st

from .theme import ACCENT


def section_header(title: str, description: str | None = None, level: int = 2) -> None:
    """Render a section header with consistent hierarchy."""

    heading = "#" * level
    st.markdown(f"{heading} {title}")
    if description:
        st.caption(description)


@contextmanager
def card(title: str | None = None, description: str | None = None) -> Iterator[None]:
    """Context manager that renders a themed card container."""

    st.markdown('<div class="valvulin-card">', unsafe_allow_html=True)
    try:
        if title:
            st.markdown(f"### {title}")
        if description:
            st.caption(description)
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def metric_columns(metrics: Iterable[Tuple[str, str | float | int]]) -> None:
    """Render metric widgets in a responsive row."""

    metrics = list(metrics)
    if not metrics:
        return

    column_count = min(4, len(metrics))
    rows = [metrics[i : i + column_count] for i in range(0, len(metrics), column_count)]
    for row in rows:
        cols = st.columns(len(row))
        for (label, value), column in zip(row, cols):
            with column:
                st.metric(label=label, value=value)


def divider() -> None:
    """Render a subtle divider consistent with the theme."""

    st.markdown('<hr class="valvulin-divider" />', unsafe_allow_html=True)


def accent_label(text: str) -> None:
    """Render a label emphasising the accent color."""

    st.markdown(f"<span style='color:{ACCENT};font-weight:600;'>{text}</span>", unsafe_allow_html=True)


__all__ = [
    "section_header",
    "card",
    "metric_columns",
    "divider",
    "accent_label",
]
