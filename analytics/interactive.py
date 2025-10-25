"""Interactive visualization helpers for OHLC data and trade annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - allow module import without pandas
    pd = None  # type: ignore

from execution.backtester import TradeRecord

from .plots import _resolve_indicator_series, _validate_ohlc


def _ensure_output_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_trade_signals_mplfinance(
    ohlc: "pd.DataFrame",
    trades: Iterable[TradeRecord],
    *,
    indicator_builders: Mapping[
        str, Callable[["pd.DataFrame"], "pd.Series | pd.DataFrame | Sequence[float]"]
    ]
    | None = None,
    mav: Sequence[int] | None = None,
    volume: bool = False,
    output_path: str | Path | None = None,
    show: bool = True,
) -> "matplotlib.figure.Figure":  # pragma: no cover - requires mplfinance
    """Render candlesticks with trade markers using :mod:`mplfinance`.

    Parameters
    ----------
    ohlc:
        Price data indexed by :class:`pandas.DatetimeIndex` containing ``open``, ``high``,
        ``low`` and ``close`` columns.
    trades:
        Iterable of :class:`execution.backtester.TradeRecord` objects to annotate.
    indicator_builders:
        Optional mapping of indicator names to callables that accept the OHLC dataframe
        and return data compatible with :func:`mplfinance.make_addplot`.
    mav:
        Moving average windows to overlay through :mod:`mplfinance`.
    volume:
        When ``True`` adds the volume panel if the dataframe contains a ``volume``
        column.
    output_path:
        Path where the resulting figure should be saved. ``.png`` and ``.html``
        extensions are supported.
    show:
        When ``True`` the matplotlib window opened by :mod:`mplfinance` is displayed.
    """

    if pd is None:
        raise RuntimeError("pandas is required to plot OHLC data with mplfinance")

    try:  # pragma: no cover - optional dependency
        import mplfinance as mpf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "mplfinance is required for plot_trade_signals_mplfinance"
        ) from exc

    cleaned = _validate_ohlc(ohlc)
    trade_list = list(trades)

    add_plots: list[Any] = []
    for label, series in _resolve_indicator_series(cleaned, indicator_builders):
        series = series.reindex(cleaned.index).astype(float)
        add_plots.append(mpf.make_addplot(series, panel=0, ylabel=label))

    entries: dict[str, pd.Series] = {}
    exits: dict[str, pd.Series] = {}

    def _series(key: str) -> pd.Series:
        if key not in entries:
            entries[key] = pd.Series(index=cleaned.index, dtype=float)
        return entries[key]

    def _exit_series(key: str) -> pd.Series:
        if key not in exits:
            exits[key] = pd.Series(index=cleaned.index, dtype=float)
        return exits[key]

    for trade in trade_list:
        entry_time = pd.Timestamp(trade.entry_time)
        exit_time = pd.Timestamp(trade.exit_time)
        pnl = float(trade.pnl)
        result = "win" if pnl > 0 else "loss" if pnl < 0 else "flat"
        side = trade.signal.side.lower()

        entry_key = f"{result}_{side}"
        exit_key = f"{result}"
        _series(entry_key).loc[entry_time] = float(trade.entry_price)
        _exit_series(exit_key).loc[exit_time] = float(trade.exit_price)

    entry_styles = {
        "win_buy": dict(marker="^", color="#2b8a3e", label="Entrada BUY (ganadora)"),
        "win_sell": dict(marker="v", color="#2b8a3e", label="Entrada SELL (ganadora)"),
        "loss_buy": dict(marker="^", color="#d9480f", label="Entrada BUY (perdedora)"),
        "loss_sell": dict(marker="v", color="#d9480f", label="Entrada SELL (perdedora)"),
        "flat_buy": dict(marker="^", color="#868e96", label="Entrada BUY (neutral)"),
        "flat_sell": dict(marker="v", color="#868e96", label="Entrada SELL (neutral)"),
    }

    exit_styles = {
        "win": dict(marker="o", color="#2b8a3e", label="Salida (ganadora)"),
        "loss": dict(marker="o", color="#d9480f", label="Salida (perdedora)"),
        "flat": dict(marker="o", color="#868e96", label="Salida (neutral)"),
    }

    for key, series in entries.items():
        series = series.reindex(cleaned.index).dropna()
        if series.empty:
            continue
        style = entry_styles[key]
        add_plots.append(
            mpf.make_addplot(
                series,
                type="scatter",
                markersize=75,
                marker=style["marker"],
                color=style["color"],
                alpha=0.9,
                label=style["label"],
            )
        )

    for key, series in exits.items():
        series = series.reindex(cleaned.index).dropna()
        if series.empty:
            continue
        style = exit_styles[key]
        add_plots.append(
            mpf.make_addplot(
                series,
                type="scatter",
                markersize=65,
                marker=style["marker"],
                color=style["color"],
                alpha=0.7,
                label=style["label"],
            )
        )

    output_path = _ensure_output_path(output_path)
    savefig = dict(fname=str(output_path)) if output_path is not None else None

    fig, _ = mpf.plot(
        cleaned,
        type="candle",
        mav=mav,
        volume=volume and "volume" in cleaned.columns,
        addplot=add_plots if add_plots else None,
        style="charles",
        tight_layout=True,
        returnfig=True,
        savefig=savefig,
        show=show,
    )

    return fig


def build_plotly_trade_figure(
    ohlc: "pd.DataFrame",
    trades: Iterable[TradeRecord],
    *,
    indicator_builders: Mapping[
        str, Callable[["pd.DataFrame"], "pd.Series | pd.DataFrame | Sequence[float]"]
    ]
    | None = None,
    title: str = "Trade Signals",
) -> "plotly.graph_objs.Figure":  # pragma: no cover - optional dependency
    """Create an interactive Plotly figure with OHLC candles and trade markers."""

    if pd is None:
        raise RuntimeError("pandas is required to build Plotly figures")

    try:  # pragma: no cover - optional dependency
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("plotly is required for build_plotly_trade_figure") from exc

    cleaned = _validate_ohlc(ohlc)
    trade_list = list(trades)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=cleaned.index,
            open=cleaned["open"],
            high=cleaned["high"],
            low=cleaned["low"],
            close=cleaned["close"],
            name="Precio",
        )
    )

    for label, series in _resolve_indicator_series(cleaned, indicator_builders):
        series = series.dropna()
        if series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=label,
            )
        )

    entry_times = []
    entry_prices = []
    entry_colors = []
    entry_symbols = []
    entry_text = []

    exit_times = []
    exit_prices = []
    exit_colors = []
    exit_text = []

    for trade in trade_list:
        pnl = float(trade.pnl)
        result = "Ganancia" if pnl > 0 else "Pérdida" if pnl < 0 else "Break-even"
        color = "#2b8a3e" if pnl > 0 else "#d9480f" if pnl < 0 else "#868e96"
        marker_symbol = "triangle-up" if trade.signal.side == "BUY" else "triangle-down"

        entry_times.append(pd.Timestamp(trade.entry_time))
        entry_prices.append(float(trade.entry_price))
        entry_colors.append(color)
        entry_symbols.append(marker_symbol)
        entry_text.append(
            f"{trade.signal.side} entry<br>Resultado: {result}<br>PnL: {pnl:+.2f}"
        )

        exit_times.append(pd.Timestamp(trade.exit_time))
        exit_prices.append(float(trade.exit_price))
        exit_colors.append(color)
        exit_text.append(
            f"Salida<br>Resultado: {result}<br>PnL: {pnl:+.2f}"
        )

    if entry_times:
        fig.add_trace(
            go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode="markers",
                marker=dict(color=entry_colors, size=10, symbol=entry_symbols, line=dict(width=1, color="black")),
                name="Entrada",
                text=entry_text,
                hoverinfo="text",
            )
        )

    if exit_times:
        fig.add_trace(
            go.Scatter(
                x=exit_times,
                y=exit_prices,
                mode="markers",
                marker=dict(color=exit_colors, size=9, symbol="x"),
                name="Salida",
                text=exit_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def show_plotly_trade_figure(
    ohlc: "pd.DataFrame",
    trades: Iterable[TradeRecord],
    *,
    indicator_builders: Mapping[
        str, Callable[["pd.DataFrame"], "pd.Series | pd.DataFrame | Sequence[float]"]
    ]
    | None = None,
    title: str = "Trade Signals",
    output_path: str | Path | None = None,
    auto_open: bool = False,
) -> "plotly.graph_objs.Figure":  # pragma: no cover - optional dependency
    """Helper to display or persist the Plotly trade figure."""

    fig = build_plotly_trade_figure(
        ohlc,
        trades,
        indicator_builders=indicator_builders,
        title=title,
    )

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.write_html(str(output_path), auto_open=auto_open)
    else:
        fig.show()

    return fig


def build_bokeh_trade_figure(
    ohlc: "pd.DataFrame",
    trades: Iterable[TradeRecord],
    *,
    indicator_builders: Mapping[
        str, Callable[["pd.DataFrame"], "pd.Series | pd.DataFrame | Sequence[float]"]
    ]
    | None = None,
    title: str = "Trade Signals",
) -> "bokeh.plotting.Figure":  # pragma: no cover - optional dependency
    """Create an interactive Bokeh figure with candles, indicators and trade markers."""

    if pd is None:
        raise RuntimeError("pandas is required to build Bokeh figures")

    try:  # pragma: no cover - optional dependency
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.plotting import figure
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("bokeh is required for build_bokeh_trade_figure") from exc

    cleaned = _validate_ohlc(ohlc)
    trade_list = list(trades)

    source = ColumnDataSource(
        data=dict(
            index=cleaned.index,
            open=cleaned["open"],
            high=cleaned["high"],
            low=cleaned["low"],
            close=cleaned["close"],
        )
    )

    inc = cleaned["close"] >= cleaned["open"]
    dec = ~inc

    diffs = cleaned.index.to_series().diff().dropna()
    spacing = diffs.min() if not diffs.empty else pd.Timedelta("1D")
    width_seconds = max(spacing.total_seconds() * 0.4, 60)

    p = figure(x_axis_type="datetime", title=title, sizing_mode="stretch_width", height=500)
    p.segment("index", "high", "index", "low", source=source, color="#4c4c4c")

    inc_source = ColumnDataSource(cleaned[inc])
    dec_source = ColumnDataSource(cleaned[dec])

    p.vbar(
        x="index",
        width=width_seconds * 1000,
        top="close",
        bottom="open",
        source=inc_source,
        fill_color="#2b8a3e",
        line_color="#2b8a3e",
    )
    p.vbar(
        x="index",
        width=width_seconds * 1000,
        top="open",
        bottom="close",
        source=dec_source,
        fill_color="#d9480f",
        line_color="#d9480f",
    )

    for label, series in _resolve_indicator_series(cleaned, indicator_builders):
        series = series.dropna()
        if series.empty:
            continue
        p.line(
            x=series.index,
            y=series.values,
            line_width=2,
            legend_label=label,
        )

    entries = []
    exits = []
    for trade in trade_list:
        pnl = float(trade.pnl)
        result = "Ganancia" if pnl > 0 else "Pérdida" if pnl < 0 else "Break-even"
        color = "#2b8a3e" if pnl > 0 else "#d9480f" if pnl < 0 else "#868e96"
        entries.append(
            dict(
                time=pd.Timestamp(trade.entry_time),
                price=float(trade.entry_price),
                side=trade.signal.side,
                result=result,
                pnl=pnl,
                color=color,
            )
        )
        exits.append(
            dict(
                time=pd.Timestamp(trade.exit_time),
                price=float(trade.exit_price),
                result=result,
                pnl=pnl,
                color=color,
            )
        )

    if entries:
        entry_source = ColumnDataSource(entries)
        p.scatter(
            x="time",
            y="price",
            source=entry_source,
            size=12,
            marker="triangle",
            color="color",
            line_color="black",
            legend_label="Entrada",
        )
        p.add_tools(
            HoverTool(
                renderers=[p.renderers[-1]],
                tooltips=[
                    ("Entrada", "@time{%F %H:%M}"),
                    ("Precio", "@price{0.0000}"),
                    ("Lado", "@side"),
                    ("Resultado", "@result"),
                    ("PnL", "@pnl{+0.00}"),
                ],
                formatters={"@time": "datetime"},
                mode="mouse",
            )
        )

    if exits:
        exit_source = ColumnDataSource(exits)
        p.scatter(
            x="time",
            y="price",
            source=exit_source,
            size=10,
            marker="x",
            color="color",
            legend_label="Salida",
        )
        p.add_tools(
            HoverTool(
                renderers=[p.renderers[-1]],
                tooltips=[
                    ("Salida", "@time{%F %H:%M}"),
                    ("Precio", "@price{0.0000}"),
                    ("Resultado", "@result"),
                    ("PnL", "@pnl{+0.00}"),
                ],
                formatters={"@time": "datetime"},
                mode="mouse",
            )
        )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def show_bokeh_trade_figure(
    ohlc: "pd.DataFrame",
    trades: Iterable[TradeRecord],
    *,
    indicator_builders: Mapping[
        str, Callable[["pd.DataFrame"], "pd.Series | pd.DataFrame | Sequence[float]"]
    ]
    | None = None,
    title: str = "Trade Signals",
    output_path: str | Path | None = None,
    show: bool = True,
) -> "bokeh.plotting.Figure":  # pragma: no cover - optional dependency
    """Display or save the Bokeh trade visualization."""

    try:  # pragma: no cover - optional dependency
        from bokeh.io import output_file, save, show as bokeh_show
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("bokeh is required for show_bokeh_trade_figure") from exc

    figure = build_bokeh_trade_figure(
        ohlc,
        trades,
        indicator_builders=indicator_builders,
        title=title,
    )

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        output_file(str(output_path))
        save(figure)
    if show:
        bokeh_show(figure)

    return figure


def launch_dash_trade_dashboard(
    ohlc: "pd.DataFrame",
    trades: Iterable[TradeRecord],
    *,
    indicator_builders: Mapping[
        str, Callable[["pd.DataFrame"], "pd.Series | pd.DataFrame | Sequence[float]"]
    ]
    | None = None,
    title: str = "Trade Signals",
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> "dash.Dash":  # pragma: no cover - optional dependency
    """Launch a minimal Dash app to explore trades interactively in the browser."""

    try:  # pragma: no cover - optional dependency
        from dash import Dash, dcc, html
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("dash is required for launch_dash_trade_dashboard") from exc

    fig = build_plotly_trade_figure(
        ohlc,
        trades,
        indicator_builders=indicator_builders,
        title=title,
    )

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2(title),
            dcc.Graph(figure=fig),
        ]
    )

    app.run(host=host, port=port, debug=debug)
    return app


__all__ = [
    "plot_trade_signals_mplfinance",
    "build_plotly_trade_figure",
    "show_plotly_trade_figure",
    "build_bokeh_trade_figure",
    "show_bokeh_trade_figure",
    "launch_dash_trade_dashboard",
]

