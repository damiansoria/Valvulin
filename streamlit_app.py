"""Interfaz Streamlit para descargar datos históricos y ejecutar backtests visuales."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from analytics.backtest_visual import run_backtest
from valvulin.data.binance_public import BinancePublicDataFeed


# Configuración inicial de la página
st.set_page_config(page_title="Valvulin Trading Bot", layout="wide")

# Inicializar el estado para los registros de log y descargas recientes
if "logs" not in st.session_state:
    st.session_state["logs"] = []

if "pending_feeds" not in st.session_state:
    st.session_state["pending_feeds"] = {}

st.title("📈 Valvulin - Bot de Trading y Backtesting")

# Navegación lateral
tab = st.sidebar.radio("📊 Sección", ["📥 Datos", "🔁 Backtesting", "⚙️ Configuración"])

if tab == "📥 Datos":
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

    if "download_logs" not in st.session_state:
        st.session_state["download_logs"] = []

    log_container = st.empty()

    def _next_widget_key(base: str) -> str:
        """Genera claves únicas y estables para widgets repetidos."""

        counter_key = f"__{base}_counter"
        counter = int(st.session_state.get(counter_key, 0))
        st.session_state[counter_key] = counter + 1
        return f"{base}_{counter}"

    def _render_download_logs() -> None:
        placeholder = log_container.empty()
        placeholder.text_area(
            "Registro de descargas",
            value="\n".join(st.session_state.get("download_logs", [])),
            height=220,
            disabled=True,
            key=_next_widget_key("download_log_area"),
        )

    _render_download_logs()

    progress_container = st.container()

    start_ts = int(
        datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000
    )

    successful_downloads: Dict[str, Path] = {}
    errors: List[str] = []

    if st.button("⬇️ Descargar / Actualizar datos históricos"):
        if not selected_symbols:
            st.warning("Debes seleccionar al menos un símbolo para descargar.")
        else:
            st.info("Iniciando proceso de descarga desde Binance...")
            st.session_state["download_logs"] = []
            _render_download_logs()

            with progress_container:
                status_box = st.status("Preparando descarga...", state="running")
                progress_bar = st.progress(0, text="Preparando descarga...")

                def safe_progress_update(pct: int, msg: str = "") -> None:
                    """Actualiza la barra de progreso sin interrumpir la app si el widget ya no existe."""

                    text = f"Descargando... {pct}% {msg}".strip()
                    try:
                        progress_bar.progress(pct, text=text)
                    except Exception:
                        pass
                    try:
                        state = "running" if pct < 100 else "complete"
                        status_box.update(label=text, state=state)
                    except Exception:
                        pass

                for symbol in selected_symbols:
                    symbol_upper = symbol.upper()
                    extension = ".csv.gz" if compress else ".csv"
                    output_path = Path(f"data/history/{symbol_upper}_{interval}{extension}")

                    def append_log(message: str) -> None:
                        st.session_state["download_logs"].append(message)
                        st.session_state["download_logs"] = st.session_state["download_logs"][-200:]
                        _render_download_logs()

                    def _chunk_callback(chunk, total_rows, sym=symbol_upper):
                        timestamp = datetime.fromtimestamp(
                            chunk.last_open_time_ms / 1000, tz=timezone.utc
                        )
                        message = (
                            f"{sym} - Bloque de {chunk.count} velas descargado. "
                            f"Última vela: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} (Total acumulado: {total_rows})"
                        )
                        append_log(message)

                    try:
                        status_box.update(
                            label=f"Descarga de {symbol_upper} ({interval}) en preparación...",
                            state="running",
                        )
                    except Exception:
                        pass
                    safe_progress_update(0, f"{symbol_upper} {interval}")
                    append_log(f"{symbol_upper} - Iniciando descarga para el intervalo {interval}.")

                    progress_callback = (
                        lambda pct, message=None, sym=symbol_upper: safe_progress_update(
                            pct,
                            " ".join(
                                part
                                for part in [f"{sym} {interval}", (message or "").strip()]
                                if part
                            ),
                        )
                    )

                    try:
                        result_path = feed.download_to_csv(
                            symbol_upper,
                            interval,
                            start_time=start_ts,
                            out_path=output_path,
                            compress=compress,
                            progress_callback=progress_callback,
                            chunk_callback=_chunk_callback,
                        )
                        successful_downloads[symbol_upper] = result_path
                        safe_progress_update(100, f"{symbol_upper} {interval} ✅")
                        append_log(f"{symbol_upper} - Descarga completada correctamente.")
                        st.success(f"✅ Datos de {symbol_upper} guardados en {result_path}")
                    except Exception as exc:  # pragma: no cover - depende de red externa
                        error_message = f"❌ Error al descargar {symbol_upper}: {exc}"
                        errors.append(error_message)
                        st.error(error_message)
                        safe_progress_update(100, f"{symbol_upper} {interval} ❌")
                        try:
                            status_box.update(
                                label=f"Descarga de {symbol_upper} ({interval}) fallida.",
                                state="error",
                            )
                        except Exception:
                            pass
                        append_log(f"{symbol_upper} - Error durante la descarga: {exc}")

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

elif tab == "🔁 Backtesting":
    st.title("🔁 Backtesting Visual")

    symbol = st.selectbox("Símbolo", ["BTCUSDT", "ETHUSDT"])
    interval = st.selectbox("Intervalo", ["1h", "4h", "1d"])

    capital_inicial = st.number_input(
        "💰 Capital inicial (USDT)", min_value=10.0, value=1000.0, step=10.0
    )
    riesgo_por_trade = st.number_input(
        "🎯 Riesgo por operación (%)", min_value=0.1, value=1.0, step=0.1
    )

    estrategias_seleccionadas = st.multiselect(
        "🧩 Estrategias a combinar",
        ["SMA Crossover", "RSI", "Bollinger Bands"],
        default=["SMA Crossover"],
    )
    logica_combinacion = st.selectbox("Lógica de combinación", ["AND", "OR"])

    strategy_params: Dict[str, Dict[str, float]] = {}
    for estrategia in estrategias_seleccionadas:
        with st.expander(f"⚙️ Parámetros {estrategia}"):
            if estrategia == "SMA Crossover":
                fast = st.number_input(
                    "SMA rápida",
                    min_value=5,
                    max_value=200,
                    value=20,
                    step=1,
                    key=f"sma_fast_{estrategia}",
                )
                slow = st.number_input(
                    "SMA lenta",
                    min_value=10,
                    max_value=300,
                    value=50,
                    step=1,
                    key=f"sma_slow_{estrategia}",
                )
                strategy_params[estrategia] = {"fast": float(fast), "slow": float(slow)}
            elif estrategia == "RSI":
                period = st.number_input(
                    "Periodo RSI",
                    min_value=5,
                    max_value=50,
                    value=14,
                    step=1,
                    key=f"rsi_period_{estrategia}",
                )
                strategy_params[estrategia] = {"period": float(period)}
            elif estrategia == "Bollinger Bands":
                period = st.number_input(
                    "Periodo",
                    min_value=5,
                    max_value=60,
                    value=20,
                    step=1,
                    key=f"bb_period_{estrategia}",
                )
                std_mult = st.number_input(
                    "Desviaciones estándar",
                    min_value=1.0,
                    max_value=3.5,
                    value=2.0,
                    step=0.1,
                    key=f"bb_std_{estrategia}",
                )
                strategy_params[estrategia] = {
                    "period": float(period),
                    "std_mult": float(std_mult),
                }

    file_path = Path(f"data/history/{symbol}_{interval}.csv")
    df = None
    if not file_path.exists():
        st.warning(
            "No hay datos descargados para este símbolo e intervalo. Descárgalos primero en la pestaña 📥 Datos."
        )
    else:
        df = pd.read_csv(file_path)
        numeric_columns = [col for col in ["open", "high", "low", "close", "volume"] if col in df.columns]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["close"], inplace=True)

    if "backtest_result" not in st.session_state:
        st.session_state["backtest_result"] = None

    if st.button("▶️ Ejecutar Backtest"):
        if df is None or df.empty:
            st.warning("Debes contar con datos históricos válidos para ejecutar el backtest.")
        elif not estrategias_seleccionadas:
            st.warning("Selecciona al menos una estrategia para continuar.")
        else:
            try:
                with st.spinner("Ejecutando backtest..."):
                    result = run_backtest(
                        df,
                        estrategias_seleccionadas,
                        strategy_params,
                        capital_inicial=capital_inicial,
                        riesgo_por_trade=riesgo_por_trade,
                        logica=logica_combinacion,
                    )
                st.session_state["backtest_result"] = {
                    "result": result,
                    "symbol": symbol,
                    "interval": interval,
                    "strategies": estrategias_seleccionadas,
                    "capital_inicial": capital_inicial,
                    "riesgo": riesgo_por_trade,
                    "logica": logica_combinacion,
                }
                st.success("✅ Backtest completado.")
            except ValueError as exc:
                st.error(f"❌ {exc}")

    stored = st.session_state.get("backtest_result")
    if stored:
        result = stored["result"]
        trades_df = result.trades.copy()
        equity_curve = result.equity_curve

        st.subheader("📈 Métricas avanzadas")
        metrics_df = pd.DataFrame([result.metrics]).T.rename(columns={0: "Valor"})
        st.dataframe(metrics_df, use_container_width=True)

        chart_df = result.data.copy()
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=chart_df["open_time"],
                    open=chart_df["open"],
                    high=chart_df["high"],
                    low=chart_df["low"],
                    close=chart_df["close"],
                    name="Velas",
                )
            ]
        )

        overlay_cols = [
            col
            for col in chart_df.columns
            if col.startswith("sma_fast_")
            or col.startswith("sma_slow_")
            or col.startswith("bb_ma_")
            or col.startswith("bb_upper_")
            or col.startswith("bb_lower_")
        ]
        for col in overlay_cols:
            fig.add_trace(
                go.Scatter(
                    x=chart_df["open_time"],
                    y=chart_df[col],
                    name=col.replace("_", " "),
                    line=dict(width=1),
                    opacity=0.6,
                )
            )

        if not trades_df.empty:
            trades_df["resultado"] = np.where(trades_df["pnl"] > 0, "Ganadora", "Perdedora")

            fig.add_trace(
                go.Scatter(
                    x=trades_df.loc[trades_df["resultado"] == "Ganadora", "entrada"],
                    y=trades_df.loc[trades_df["resultado"] == "Ganadora", "precio_entrada"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="#2ecc71"),
                    name="Entrada ganadora",
                    hovertemplate=
                    "Entrada: %{x}<br>Precio: %{y}<br>PNL %: %{customdata:.2%}<extra></extra>",
                    customdata=trades_df.loc[trades_df["resultado"] == "Ganadora", "pnl"],
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=trades_df.loc[trades_df["resultado"] == "Perdedora", "entrada"],
                    y=trades_df.loc[trades_df["resultado"] == "Perdedora", "precio_entrada"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="#e74c3c"),
                    name="Entrada perdedora",
                    hovertemplate=
                    "Entrada: %{x}<br>Precio: %{y}<br>PNL %: %{customdata:.2%}<extra></extra>",
                    customdata=trades_df.loc[trades_df["resultado"] == "Perdedora", "pnl"],
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=trades_df["salida"],
                    y=trades_df["precio_salida"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="#3498db"),
                    name="Salida",
                    hovertemplate=
                    "Salida: %{x}<br>Precio: %{y}<br>PNL %: %{customdata:.2%}<extra></extra>",
                    customdata=trades_df["pnl"],
                )
            )

        fig.update_layout(title="📉 Señales sobre el precio", legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📈 Curva de Patrimonio")
        st.line_chart(equity_curve, height=300)

        st.subheader("📉 Curva de Drawdown")
        drawdown_chart = result.drawdown * 100 if hasattr(result, "drawdown") else None
        if drawdown_chart is not None:
            st.line_chart(drawdown_chart.rename("Drawdown %"), height=200)

        if not trades_df.empty:
            st.subheader("📊 Distribución de Retornos")
            hist_df = trades_df.copy()
            hist_df["resultado"] = np.where(hist_df["pnl"] > 0, "Ganadora", "Perdedora")
            fig_hist = px.histogram(
                hist_df,
                x="pnl",
                nbins=20,
                color="resultado",
                color_discrete_map={"Ganadora": "#2ecc71", "Perdedora": "#e74c3c"},
                labels={"pnl": "Retorno por operación"},
            )
            fig_hist.update_layout(bargap=0.15)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("🧾 Operaciones")
        trades_display = trades_df.copy()
        if not trades_display.empty:
            trades_display["pnl_%"] = (trades_display["pnl"] * 100).round(4)
            trades_display["entrada"] = trades_display["entrada"].dt.strftime("%Y-%m-%d %H:%M")
            trades_display["salida"] = trades_display["salida"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(trades_display, use_container_width=True)
        else:
            st.info("No se generaron operaciones con la configuración actual.")

        export_col1, export_col2 = st.columns([0.2, 0.8])
        with export_col1:
            if st.button("💾 Exportar resultados", key="export_backtest"):
                if trades_df.empty:
                    st.warning("No hay operaciones para exportar en CSV.")
                else:
                    output_dir = Path("data/backtests")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    strategy_label = "+".join(stored["strategies"]).replace(" ", "")
                    base_name = f"{stored['symbol']}_{strategy_label}_{stored['interval']}"

                    trades_path = output_dir / f"{base_name}_trades.csv"
                    trades_df.to_csv(trades_path, index=False)

                    equity_curve = result.equity_curve
                    drawdown_curve = result.drawdown.reindex(
                        equity_curve.index, fill_value=0.0
                    )
                    equity_export = equity_curve.reset_index()
                    equity_export.columns = ["timestamp", "equity"]
                    equity_export["drawdown"] = drawdown_curve.reset_index(drop=True)
                    equity_path = output_dir / f"{base_name}_equity_curve.csv"
                    equity_export.to_csv(equity_path, index=False)

                    metrics_payload: Dict[str, float | int | None] = {}
                    for key, value in result.metrics.items():
                        converted = value
                        if isinstance(converted, (np.floating, np.integer)):
                            converted = float(converted)
                        if isinstance(converted, (float, int)):
                            if isinstance(converted, float) and not np.isfinite(converted):
                                metrics_payload[key] = None
                            else:
                                metrics_payload[key] = converted
                        else:
                            metrics_payload[key] = converted

                    metrics_path = output_dir / f"{base_name}_metrics.json"
                    metrics_path.write_text(
                        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                    st.success(
                        "Resultados exportados correctamente (operaciones, equity y métricas)."
                    )

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
