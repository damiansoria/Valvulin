"""Utility modules for analyzing trading performance and logging activity."""

from .trade_logger import TradeEntry, TradeLogger
from .performance import PerformanceMetrics, compute_performance_metrics

__all__ = [
    "TradeEntry",
    "TradeLogger",
    "PerformanceMetrics",
    "compute_performance_metrics",
]

try:  # pragma: no cover - optional interactive exports
    from .interactive import (
        build_bokeh_trade_figure,
        build_plotly_trade_figure,
        launch_dash_trade_dashboard,
        plot_trade_signals_mplfinance,
        show_bokeh_trade_figure,
        show_plotly_trade_figure,
    )

    __all__ += [
        "build_bokeh_trade_figure",
        "build_plotly_trade_figure",
        "launch_dash_trade_dashboard",
        "plot_trade_signals_mplfinance",
        "show_bokeh_trade_figure",
        "show_plotly_trade_figure",
    ]
except Exception:  # pragma: no cover - optional dependencies missing
    pass
