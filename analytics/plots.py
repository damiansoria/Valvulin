"""Plotting helpers for trading performance analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - keep working even if pandas is unavailable
    pd = None  # type: ignore

from .performance import group_metrics_by_strategy
from .trade_logger import TradeEntry
from execution.backtester import TradeRecord


def _ensure_output_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_equity_curve(
    trades: Iterable[TradeEntry],
    output_path: Path | str | None = None,
    show: bool = False,
):
    """Plot an equity curve using cumulative R multiples."""

    trade_list = list(trades)
    cumulative = []
    running = 0.0
    for trade in trade_list:
        running += trade.r_multiple
        cumulative.append(running)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative, marker="o")
    ax.set_title("Equity Curve (R multiples)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative R")
    ax.grid(True, linestyle="--", alpha=0.4)

    output_path = _ensure_output_path(output_path)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_r_distribution(
    trades: Iterable[TradeEntry],
    bins: int = 20,
    output_path: Path | str | None = None,
    show: bool = False,
):
    """Plot histogram of trade R multiples."""

    r_values = [trade.r_multiple for trade in trades]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(r_values, bins=bins, color="#2b8a3e", alpha=0.7, edgecolor="black")
    ax.set_title("Distribución de R multiples")
    ax.set_xlabel("R multiple")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path = _ensure_output_path(output_path)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_strategy_patterns(
    trades: Iterable[TradeEntry],
    output_path: Path | str | None = None,
    show: bool = False,
):
    """Plot expectancy by strategy to highlight behavioural patterns."""

    trade_list = list(trades)
    metrics = group_metrics_by_strategy(trade_list)
    if not metrics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No hay datos", ha="center", va="center")
        ax.axis("off")
    else:
        labels = list(metrics.keys())
        expectancy = [m.expectancy for m in metrics.values()]
        win_rates = [m.win_rate for m in metrics.values()]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, expectancy, color="#1c7ed6", alpha=0.8)
        ax.set_title("Expectancy por estrategia")
        ax.set_ylabel("Expectancy (R)")
        ax.set_xlabel("Estrategia")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(bottom=min(0, min(expectancy)))

        for bar, rate in zip(bars, win_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"Win Rate: {rate:.1%}",
                ha="center",
                va="bottom",
            )

    output_path = _ensure_output_path(output_path)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _validate_ohlc(ohlc: "pd.DataFrame") -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required to plot OHLC data")
    if not isinstance(ohlc, pd.DataFrame):  # type: ignore[unreachable]
        raise TypeError("ohlc must be a pandas.DataFrame")
    required = {"open", "high", "low", "close"}
    missing = required - set(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC data missing required columns: {sorted(missing)}")
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise TypeError("OHLC data must be indexed by pandas.DatetimeIndex")
    return ohlc.sort_index()


def _resolve_indicator_series(
    ohlc: "pd.DataFrame",
    indicator_builders: Mapping[str, Callable[["pd.DataFrame"], "pd.Series | pd.DataFrame | Sequence[float]"]]
    | None,
) -> list[tuple[str, "pd.Series"]]:
    if indicator_builders is None:
        return []

    resolved: list[tuple[str, "pd.Series"]] = []
    for name, builder in indicator_builders.items():
        result = builder(ohlc)
        if isinstance(result, pd.Series):
            resolved.append((name, result))
        elif isinstance(result, pd.DataFrame):
            for column in result.columns:
                resolved.append((f"{name} ({column})", result[column]))
        else:
            series = pd.Series(result, index=ohlc.index, name=name)  # type: ignore[arg-type]
            resolved.append((name, series))
    return resolved


def _plot_candlesticks(ax: plt.Axes, ohlc: "pd.DataFrame") -> None:
    ohlc_numeric = ohlc[["open", "high", "low", "close"]].copy()
    ohlc_numeric["timestamp"] = mdates.date2num(ohlc.index.to_pydatetime())
    if len(ohlc_numeric) > 1:
        spacing = ohlc_numeric["timestamp"].diff().dropna().min()
        width = float(spacing) * 0.6 if spacing and spacing > 0 else 0.6
    else:
        width = 0.6
    width = max(width, 0.05)

    for _, row in ohlc_numeric.iterrows():
        color = "#2b8a3e" if row["close"] >= row["open"] else "#d9480f"
        ax.vlines(row["timestamp"], row["low"], row["high"], color=color, linewidth=1.0, alpha=0.9)
        body_height = abs(row["close"] - row["open"])
        body_height = body_height if body_height > 0 else max(row["high"] - row["low"], 1e-6) * 0.02
        lower = min(row["open"], row["close"])
        candle = Rectangle(
            (row["timestamp"] - width / 2, lower),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax.add_patch(candle)

    ax.set_xlim(ohlc_numeric["timestamp"].min() - width, ohlc_numeric["timestamp"].max() + width)
    ax.set_ylabel("Price")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))


def plot_trade_signals(
    ohlc: "pd.DataFrame",
    trades: Iterable[TradeRecord],
    *,
    indicator_builders: Mapping[
        str, Callable[["pd.DataFrame"], "pd.Series | pd.DataFrame | Sequence[float]"]
    ]
    | None = None,
    output_path: Path | str | None = None,
    show: bool = False,
    annotate_exits: bool = True,
) -> plt.Figure:
    """Plot candlesticks with indicator overlays and trade markers.

    Parameters
    ----------
    ohlc:
        Price data indexed by :class:`pandas.DatetimeIndex` containing ``open``, ``high``,
        ``low`` and ``close`` columns.
    trades:
        Iterable of :class:`execution.backtester.TradeRecord` objects to annotate.
    indicator_builders:
        Mapping of indicator names to callables that receive the OHLC dataframe and return
        either a :class:`pandas.Series`, :class:`pandas.DataFrame` or a sequence of values.
        The resulting series are plotted on top of the candlesticks. Examples include EMA
        or MACD calculations derived from the provided OHLC data.
    output_path:
        Optional path to save the generated figure.
    show:
        Whether to call :func:`matplotlib.pyplot.show` once the plot is generated.
    annotate_exits:
        When ``True`` add exit markers and stop-loss annotations for each trade.
    """

    if pd is None:
        raise RuntimeError("pandas is required to use plot_trade_signals")

    cleaned = _validate_ohlc(ohlc)
    trade_list = list(trades)

    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_candlesticks(ax, cleaned)

    for label, series in _resolve_indicator_series(cleaned, indicator_builders):
        series = series.dropna()
        if series.empty:
            continue
        ax.plot(series.index, series.values, label=label, linewidth=1.3)

    legend_labels: list[str] = []
    legend_seen: set[str] = set()

    for trade in trade_list:
        entry_time = trade.entry_time
        exit_time = trade.exit_time
        entry_price = trade.entry_price
        exit_price = trade.exit_price
        side = trade.signal.side
        pnl = trade.pnl
        if not isinstance(entry_time, pd.Timestamp):
            entry_time = pd.Timestamp(entry_time)
        if not isinstance(exit_time, pd.Timestamp):
            exit_time = pd.Timestamp(exit_time)

        color = "#2b8a3e" if pnl > 0 else "#d9480f" if pnl < 0 else "#868e96"
        marker = "^" if side == "BUY" else "v"
        result_label = "Profitable" if pnl > 0 else "Losing" if pnl < 0 else "Break-even"
        entry_label = f"{result_label} {side.title()} entry"
        label = entry_label if entry_label not in legend_seen else None
        ax.scatter(
            entry_time,
            entry_price,
            color=color,
            marker=marker,
            s=70,
            edgecolors="black",
            linewidths=0.6,
            label=label,
            zorder=5,
        )
        if entry_label not in legend_seen:
            legend_labels.append(entry_label)
            legend_seen.add(entry_label)

        if annotate_exits:
            exit_label = "Trade exit" if "Trade exit" not in legend_seen else None
            ax.scatter(
                exit_time,
                exit_price,
                color=color,
                marker="x",
                s=60,
                linewidths=1.2,
                label=exit_label,
                zorder=5,
            )
            if "Trade exit" not in legend_seen:
                legend_labels.append("Trade exit")
                legend_seen.add("Trade exit")
            if trade.signal.stop_loss is not None:
                stop_label = "Stop loss" if "Stop loss" not in legend_seen else None
                ax.hlines(
                    trade.signal.stop_loss,
                    xmin=entry_time,
                    xmax=exit_time,
                    colors=color,
                    linestyles="dotted",
                    linewidth=1.0,
                    label=stop_label,
                    alpha=0.4,
                )
                if "Stop loss" not in legend_seen:
                    legend_labels.append("Stop loss")
                    legend_seen.add("Stop loss")
            ax.text(
                exit_time,
                exit_price,
                f"PnL: {pnl:+.2f}",
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
            )

    if legend_labels:
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            label_to_handle[label] = handle
        manual_handles = [label_to_handle[label] for label in legend_labels if label in label_to_handle]
        manual_labels = [label for label in legend_labels if label in label_to_handle]
        indicator_handles: list[Any] = []
        indicator_labels: list[str] = []
        for handle, label in zip(handles, labels):
            if label not in legend_seen:
                indicator_handles.append(handle)
                indicator_labels.append(label)
        combined_handles = manual_handles + indicator_handles
        combined_labels = manual_labels + indicator_labels
        if combined_handles:
            ax.legend(combined_handles, combined_labels, loc="best", fontsize=9)

    ax.set_title("Trade Signals")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


__all__ = [
    "plot_equity_curve",
    "plot_r_distribution",
    "plot_strategy_patterns",
    "plot_trade_signals",
]
