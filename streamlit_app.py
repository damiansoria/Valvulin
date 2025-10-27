"""Interfaz Streamlit para descargar datos históricos y ejecutar backtests visuales."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from analytics.backtest_visual import run_backtest
from valvulin.data.binance_public import BinancePublicDataFeed


def _list_history_symbols() -> Dict[str, List[str]]:
    """Obtiene un diccionario {símbolo: [intervalos disponibles]} a partir de los archivos en disco."""

    data_dir = Path("data/history")
    symbols: Dict[str, set[str]] = {}
    if not data_dir.exists():
        return {}

    for file_path in data_dir.glob("*.csv*"):
        parts = file_path.name.split("_")
        if len(parts) < 2:
            continue
        symbol = parts[0]
        interval = parts[1].split(".")[0]
        symbols.setdefault(symbol, set()).add(interval)

    return {symbol: sorted(intervals) for symbol, intervals in symbols.items()}


def _resolve_history_path(symbol: str, interval: str) -> Path | None:
    """Devuelve la ruta del CSV (o CSV.GZ) para un símbolo e intervalo dado."""

    data_dir = Path("data/history")
    gz_path = data_dir / f"{symbol}_{interval}.csv.gz"
    csv_path = data_dir / f"{symbol}_{interval}.csv"
    if gz_path.exists():
        return gz_path
    if csv_path.exists():
        return csv_path
    return None


def _load_history_dataframe(path: Path) -> pd.DataFrame:
    """Carga un archivo CSV de datos históricos en un DataFrame."""

    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, compression=compression)


def _format_percentage(value: float | None) -> str:
    """Formatea un valor porcentual manejando NaN o None."""

    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:.2f}%"


def _format_ratio(value: float | None) -> str:
    """Formatea un ratio como el profit factor."""

    if value is None or pd.isna(value):
        return "N/D"
    if value == float("inf"):
        return "∞"
    return f"{value:.2f}"


def _format_integer(value: float | None) -> str:
    """Formatea un entero a texto."""

    if value is None or pd.isna(value):
        return "N/D"
    return f"{int(value)}"


def _create_backtest_chart(processed: pd.DataFrame, trades: pd.DataFrame, strategy: str) -> go.Figure:
    """Construye el gráfico interactivo con velas, indicadores y señales."""

    data = processed.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data.dropna(subset=["timestamp"], inplace=True)

    rows = 2 if strategy == "RSI Oversold/Overbought" else 1
    row_heights = [0.7, 0.3] if rows == 2 else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

    fig.add_trace(
        go.Candlestick(
            x=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Precio",
            increasing_line_color="#089981",
            decreasing_line_color="#f23645",
        ),
        row=1,
        col=1,
    )

    if strategy == "SMA Crossover":
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data.get("sma_fast"),
                mode="lines",
                name="SMA rápida",
                line=dict(color="#00c3ff", width=1.8),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data.get("sma_slow"),
                mode="lines",
                name="SMA lenta",
                line=dict(color="#f1c40f", width=1.8),
            ),
            row=1,
            col=1,
        )
    elif strategy == "Bollinger Bands Reversal":
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data.get("bb_upper"),
                mode="lines",
                name="Banda superior",
                line=dict(color="#ff6b6b", width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data.get("bb_mid"),
                mode="lines",
                name="Media",
                line=dict(color="#3498db", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data.get("bb_lower"),
                mode="lines",
                name="Banda inferior",
                line=dict(color="#2ecc71", width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )
    elif strategy == "RSI Oversold/Overbought":
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data.get("rsi"),
                mode="lines",
                name="RSI",
                line=dict(color="#9b59b6", width=1.8),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line=dict(color="#e74c3c", width=1, dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#2ecc71", width=1, dash="dash"), row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    if not trades.empty:
        entries = trades.copy()
        entries["entry_time"] = pd.to_datetime(entries["entry_time"], utc=True)
        entries.dropna(subset=["entry_time"], inplace=True)
        exits = trades.copy()
        exits["exit_time"] = pd.to_datetime(exits["exit_time"], utc=True)
        exits.dropna(subset=["exit_time"], inplace=True)

        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries["entry_time"],
                    y=entries["entry_price"],
                    mode="markers",
                    name="Entradas",
                    marker=dict(symbol="triangle-up", size=12, color="#00cc96"),
                ),
                row=1,
                col=1,
            )
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits["exit_time"],
                    y=exits["exit_price"],
                    mode="markers",
                    name="Salidas",
                    marker=dict(symbol="triangle-down", size=12, color="#ff3b30"),
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_xaxes(showgrid=True, row=rows, col=1)
    return fig


# Configuración inicial de la página
st.set_page_config(page_title="Valvulin Trading Bot", layout="wide")

# Inicializar el estado para los registros de log y descargas recientes
if "logs" not in st.session_state:
    st.session_state["logs"] = []

if "pending_feeds" not in st.session_state:
    st.session_state["pending_feeds"] = {}

st.title("📈 Valvulin - Bot de Trading y Backtesting")

# Navegación lateral
current_tab = st.sidebar.radio("📊 Sección", ["📥 Datos", "🔁 Backtesting", "⚙️ Configuración"])

if current_tab == "📥 Datos":
    st.sidebar.header("⚙️ Configuración de datos")
    user_symbol = st.sidebar.text_input("Símbolo principal", value="BTCUSDT").upper().strip()
    predefined_symbols: List[str] = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
    ]

    if user_symbol:
        predefined_symbols = sorted(set(predefined_symbols + [user_symbol]), key=lambda item: item)

    selected_symbols = st.sidebar.multiselect(
        "Selecciona los símbolos a descargar",
        options=predefined_symbols,
        default=[user_symbol] if user_symbol else [predefined_symbols[0]],
    )

    interval = st.sidebar.selectbox("Intervalo", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    start_date = st.sidebar.date_input("Fecha de inicio", datetime(2024, 1, 1))
    compress = st.sidebar.checkbox("Guardar comprimido (.csv.gz)", value=False)

    st.sidebar.divider()
    advanced_mode = st.sidebar.toggle("Modo Avanzado", value=False)

    if advanced_mode:
        limit_value = st.sidebar.number_input(
            "Número máximo de velas por request",
            min_value=100,
            max_value=1000,
            value=1000,
            step=50,
        )
        rate_sleep = st.sidebar.number_input(
            "Intervalo entre requests (segundos)",
            min_value=0.0,
            max_value=5.0,
            value=0.35,
            step=0.05,
            format="%.2f",
        )
        continue_download = st.sidebar.checkbox("Continuar descarga desde último punto disponible", value=True)
    else:
        limit_value = 1000
        rate_sleep = 0.35
        continue_download = True

    feed = BinancePublicDataFeed(
        rate_sleep=rate_sleep,
        compress=compress,
        cache=continue_download,
        max_limit=int(limit_value),
    )

    st.write("### Descargar o actualizar datos históricos")
    st.info("Puedes usar esta herramienta para descargar datos públicos de Binance sin necesidad de API Key.")

    config_path = Path("config.yaml")
    config_data: Dict[str, Iterable] | None = None
    if config_path.exists():
        try:
            config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            st.warning(f"⚠️ No se pudo leer config.yaml: {exc}")

    if config_data and isinstance(config_data, dict) and config_data.get("data_feeds"):
        feed_rows = []
        for feed_cfg in config_data.get("data_feeds", []):
            if not isinstance(feed_cfg, dict):
                continue
            params = feed_cfg.get("params") or {}
            feed_rows.append(
                {
                    "Nombre": feed_cfg.get("name", ""),
                    "Tipo": feed_cfg.get("type", ""),
                    "Archivo": params.get("file_path", ""),
                }
            )
        if feed_rows:
            st.write("#### Feeds configurados en config.yaml")
            st.table(pd.DataFrame(feed_rows))

    status_container = st.container()
    now_utc = datetime.now(timezone.utc)

    with status_container:
        if not selected_symbols:
            st.warning("Selecciona al menos un símbolo para continuar.")
        else:
            for symbol in selected_symbols:
                plain_file = Path(f"data/history/{symbol}_{interval}.csv")
                gz_file = Path(f"data/history/{symbol}_{interval}.csv.gz")
                target_file = gz_file if gz_file.exists() else plain_file

                if target_file.exists():
                    file_mtime = datetime.fromtimestamp(target_file.stat().st_mtime, tz=timezone.utc)
                    file_age = now_utc - file_mtime
                    last_update = file_mtime.strftime("%Y-%m-%d %H:%M:%S UTC")
                    size_mb = round(target_file.stat().st_size / 1024 / 1024, 2)
                    if file_age <= timedelta(hours=12):
                        st.success(
                            f"{symbol} ({interval}) - ✅ Los datos están actualizados al último día disponible.\n"
                            f"📁 Última actualización: {last_update} | 🧾 Tamaño: {size_mb} MB"
                        )
                    else:
                        st.info(
                            f"{symbol} ({interval}) - Datos disponibles pero desactualizados.\n"
                            f"📁 Última actualización: {last_update} | 🧾 Tamaño: {size_mb} MB"
                        )
                else:
                    st.warning(f"{symbol} ({interval}) - ⚠️ Aún no se ha descargado ningún archivo de datos.")

    log_placeholder = st.empty()
    log_placeholder.text_area(
        "Registro de descargas",
        value="\n".join(st.session_state.get("logs", [])),
        height=220,
        disabled=True,
    )

    start_ts = int(
        datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000
    )

    successful_downloads: Dict[str, Path] = {}
    errors: List[str] = []

    if st.button("⬇️ Descargar / Actualizar datos históricos"):
        if not selected_symbols:
            st.warning("Debes seleccionar al menos un símbolo para descargar.")
        else:
            st.session_state["logs"] = []
            log_placeholder.text_area("Registro de descargas", value="", height=220, disabled=True)

            for symbol in selected_symbols:
                symbol_upper = symbol.upper()
                extension = ".csv.gz" if compress else ".csv"
                output_path = Path(f"data/history/{symbol_upper}_{interval}{extension}")

                def _progress_callback(chunk, total_rows, sym=symbol_upper):
                    timestamp = datetime.fromtimestamp(chunk.last_open_time_ms / 1000, tz=timezone.utc)
                    message = (
                        f"{sym} - Bloque de {chunk.count} velas descargado. "
                        f"Última vela: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} (Total: {total_rows})"
                    )
                    st.session_state["logs"].append(message)
                    st.session_state["logs"] = st.session_state["logs"][-200:]
                    log_placeholder.text_area(
                        "Registro de descargas",
                        value="\n".join(st.session_state["logs"]),
                        height=220,
                        disabled=True,
                    )

                with st.spinner(f"Descargando datos para {symbol_upper} ({interval})..."):
                    try:
                        result_path = feed.download_to_csv(
                            symbol_upper,
                            interval,
                            start_time=start_ts,
                            out_path=output_path,
                            compress=compress,
                            progress_callback=_progress_callback,
                        )
                        successful_downloads[symbol_upper] = result_path
                        st.success(f"✅ Datos de {symbol_upper} guardados en {result_path}")
                    except Exception as exc:  # pragma: no cover - depende de red externa
                        error_message = f"❌ Error al descargar {symbol_upper}: {exc}"
                        errors.append(error_message)
                        st.error(error_message)

            if errors:
                st.warning("Se encontraron problemas con algunos símbolos. Revisa los mensajes anteriores.")

    if successful_downloads:
        st.session_state["pending_feeds"] = {
            symbol: path for symbol, path in successful_downloads.items()
        }

    if st.session_state.get("pending_feeds"):
        new_feeds = st.session_state["pending_feeds"]
        st.write("---")
        st.write("### Integrar descargas en config.yaml")
        st.write(
            "Puedes agregar automáticamente los nuevos archivos descargados al archivo `config.yaml` para usarlos en tus backtests."
        )

        if st.button("Actualizar config.yaml con los nuevos feeds CSV"):
            updated = False
            config_to_update: Dict[str, Iterable] = {}
            if config_path.exists():
                try:
                    config_to_update = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                except yaml.YAMLError as exc:
                    st.error(f"No se pudo procesar config.yaml: {exc}")
                    config_to_update = {}
            feeds_list = list(config_to_update.get("data_feeds", []))
            existing_names = {feed.get("name") for feed in feeds_list if isinstance(feed, dict)}

            for sym, path in new_feeds.items():
                feed_name = f"{sym}-{interval}"
                if feed_name in existing_names:
                    continue
                feeds_list.append(
                    {
                        "name": feed_name,
                        "type": "csv",
                        "params": {"file_path": path.as_posix()},
                    }
                )
                existing_names.add(feed_name)
                updated = True

            if updated:
                config_to_update["data_feeds"] = feeds_list
                with config_path.open("w", encoding="utf-8") as config_file:
                    yaml.safe_dump(config_to_update, config_file, sort_keys=False, allow_unicode=True)
                st.success("config.yaml actualizado correctamente.")
            else:
                st.info("No se realizaron cambios porque los feeds ya estaban registrados.")

            st.session_state["pending_feeds"] = {}

    if errors:
        for error in errors:
            st.error(error)

elif current_tab == "🔁 Backtesting":
    st.write("### 🔁 Backtesting visual")
    st.info(
        "Selecciona el activo, la estrategia y los parámetros para ejecutar un backtest interactivo sobre tus datos históricos."
    )

    available_map = _list_history_symbols()
    default_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
    ]
    symbol_options = sorted(set(list(available_map.keys()) + default_symbols)) or default_symbols

    today = datetime.now(timezone.utc).date()
    default_start = today - timedelta(days=90)
    strategy_options = [
        "SMA Crossover",
        "RSI Oversold/Overbought",
        "Bollinger Bands Reversal",
    ]

    with st.form("backtest_form"):
        st.subheader("Configura tu simulación")
        col_symbol, col_interval, col_strategy = st.columns(3)
        selected_symbol = col_symbol.selectbox("Símbolo", symbol_options, index=0)
        interval_choices = available_map.get(selected_symbol, ["1m", "5m", "15m", "1h", "4h", "1d"])
        default_interval_index = interval_choices.index("1h") if "1h" in interval_choices else 0
        selected_interval = col_interval.selectbox("Intervalo", interval_choices, index=default_interval_index)
        selected_strategy = col_strategy.selectbox("Estrategia", strategy_options)

        col_start, col_end, _ = st.columns(3)
        selected_start = col_start.date_input("Fecha inicio", default_start)
        selected_end = col_end.date_input("Fecha fin", today)

        st.write("#### Parámetros de la estrategia")
        params: Dict[str, float] = {}
        if selected_strategy == "SMA Crossover":
            fast_col, slow_col = st.columns(2)
            params["sma_fast"] = fast_col.number_input("SMA rápida", min_value=2, max_value=200, value=20)
            params["sma_slow"] = slow_col.number_input("SMA lenta", min_value=3, max_value=400, value=50)
        elif selected_strategy == "RSI Oversold/Overbought":
            period_col, lower_col, upper_col = st.columns(3)
            params["rsi_period"] = period_col.number_input("Periodo RSI", min_value=2, max_value=100, value=14)
            params["rsi_lower"] = lower_col.number_input("Umbral sobreventa", min_value=5, max_value=50, value=30)
            params["rsi_upper"] = upper_col.number_input("Umbral sobrecompra", min_value=50, max_value=95, value=70)
        elif selected_strategy == "Bollinger Bands Reversal":
            window_col, std_col = st.columns(2)
            params["bb_window"] = window_col.number_input("Ventana", min_value=5, max_value=200, value=20)
            params["bb_std"] = std_col.number_input("Desviaciones estándar", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

        submitted = st.form_submit_button("▶️ Ejecutar Backtest")

    if submitted:
        if selected_start > selected_end:
            st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        else:
            history_path = _resolve_history_path(selected_symbol, selected_interval)
            if history_path is None:
                st.warning(
                    "No se encontró un archivo de datos para el símbolo e intervalo seleccionados. Descarga primero el histórico desde la pestaña de Datos."
                )
            else:
                try:
                    raw_data = _load_history_dataframe(history_path)
                except Exception as exc:  # pragma: no cover - lectura de archivos externos
                    st.error(f"No se pudo cargar el archivo de datos: {exc}")
                    raw_data = pd.DataFrame()

                if not raw_data.empty:
                    raw_data["open_time"] = pd.to_datetime(raw_data["open_time"], utc=True, errors="coerce")
                    raw_data.dropna(subset=["open_time"], inplace=True)
                    start_dt = datetime.combine(selected_start, datetime.min.time()).replace(tzinfo=timezone.utc)
                    end_dt = datetime.combine(selected_end, datetime.max.time()).replace(tzinfo=timezone.utc)
                    filtered = raw_data[(raw_data["open_time"] >= start_dt) & (raw_data["open_time"] <= end_dt)].copy()

                    if filtered.empty:
                        st.warning("El rango seleccionado no contiene datos. Ajusta las fechas o descarga más histórico.")
                    else:
                        with st.spinner("Ejecutando backtest..."):
                            try:
                                result = run_backtest(filtered, selected_strategy, params)
                            except Exception as exc:
                                st.error(f"Ocurrió un error durante el backtest: {exc}")
                                result = None

                        if result:
                            trades_df: pd.DataFrame = result.get("trades", pd.DataFrame())
                            metrics: Dict[str, float] = result.get("metrics", {})
                            equity_curve: pd.DataFrame = result.get("equity_curve", pd.DataFrame())
                            processed = result.get("processed_data", filtered)

                            st.write("### 📊 Métricas del backtest")
                            metric_cols = st.columns(5)
                            metric_cols[0].metric("Ganancia acumulada", _format_percentage(metrics.get("total_return_pct")))
                            metric_cols[1].metric("Profit factor", _format_ratio(metrics.get("profit_factor")))
                            metric_cols[2].metric("Winrate", _format_percentage(metrics.get("winrate")))
                            metric_cols[3].metric("Operaciones", _format_integer(metrics.get("num_trades")))
                            metric_cols[4].metric("Drawdown máximo", _format_percentage(metrics.get("max_drawdown_pct")))

                            st.write("### 📈 Gráfico interactivo")
                            chart = _create_backtest_chart(processed, trades_df, selected_strategy)
                            st.plotly_chart(chart, use_container_width=True)

                            if not equity_curve.empty:
                                st.write("### 📉 Curva de equity")
                                equity_curve_sorted = equity_curve.copy()
                                equity_curve_sorted["timestamp"] = pd.to_datetime(
                                    equity_curve_sorted["timestamp"], utc=True, errors="coerce"
                                )
                                equity_curve_sorted.dropna(subset=["timestamp"], inplace=True)
                                st.line_chart(
                                    equity_curve_sorted.set_index("timestamp")["equity"],
                                    use_container_width=True,
                                )

                            st.write("### 🧾 Operaciones ejecutadas")
                            if not trades_df.empty:
                                trades_to_show = trades_df.copy()
                                trades_to_show["entry_time"] = pd.to_datetime(trades_to_show["entry_time"], utc=True)
                                trades_to_show["exit_time"] = pd.to_datetime(trades_to_show["exit_time"], utc=True)
                                trades_to_show["return_pct"] = trades_to_show["return_pct"].round(2)
                                trades_to_show["duracion_min"] = trades_to_show["duracion_min"].round(2)
                                st.dataframe(trades_to_show, use_container_width=True)

                                csv_data = trades_to_show.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "💾 Descargar operaciones en CSV",
                                    data=csv_data,
                                    file_name=f"{selected_symbol}_{selected_interval}_{selected_strategy.replace(' ', '_')}.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.info("La estrategia no generó operaciones en el periodo seleccionado.")
                else:
                    st.warning("El archivo de datos está vacío o no se pudo leer correctamente.")

else:
    st.write("### ⚙️ Configuración y ayuda")
    st.write(
        "Utiliza esta sección para gestionar archivos de configuración avanzados, documentar tus estrategias o añadir notas de operación."
    )
    st.write(
        "- Descarga datos desde la pestaña **Datos**.\n"
        "- Ejecuta backtests visuales en la pestaña **Backtesting**.\n"
        "- Personaliza tus `config.yaml` con los botones disponibles tras cada descarga."
    )
