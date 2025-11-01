"""Interfaz Streamlit para descargar datos históricos y ejecutar backtests visuales."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from analytics.backtest_visual import run_backtest as run_visual_backtest
from analytics.backtest_storage import (
    list_backtest_runs,
    load_backtest_run,
    save_backtest_run,
)
from analytics.csv_normalization import normalize_trade_dataframe
from analytics.metrics import compute_backtest_summary_metrics, prepare_drawdown_columns
from analytics.plot_utils import (
    plot_equity_curve,
    plot_drawdown_curve,
    plot_expectancy_bar,
    plot_r_multiple_distribution,
)
from valvulin.data.binance_public import BinancePublicDataFeed
from optimization import run_hybrid_optimization
from optimization.bayesian_optimizer import BOUNDS as HYBRID_BOUNDS
from modules.backtesting import run_backtest
from utils.pdf_report import generate_pdf_report


# Configuración inicial de la página
st.set_page_config(page_title="Valvulin Trading Bot", layout="wide")

# Inicializar el estado para los registros de log y descargas recientes
if "logs" not in st.session_state:
    st.session_state["logs"] = []

if "pending_feeds" not in st.session_state:
    st.session_state["pending_feeds"] = {}

# Navegación lateral
tab = st.sidebar.radio(
    "📊 Sección", ["📥 Datos", "🔁 Backtesting", "📊 Analytics", "⚙️ Configuración"]
)

if tab == "📥 Datos":
    st.title("📈 Valvulin - Bot de Trading y Backtesting")
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
    st.title("🔁 Valvulin - Backtesting")

    optimizer_defaults = st.session_state.get("advanced_optimizer_best_params")
    if "backtest_sl_ratio" not in st.session_state:
        if optimizer_defaults:
            st.session_state["backtest_sl_ratio"] = float(
                np.clip(
                    optimizer_defaults.get("atr_mult_sl", 1.0),
                    0.1,
                    10.0,
                )
            )
        else:
            st.session_state["backtest_sl_ratio"] = 1.0
    if "backtest_tp_ratio" not in st.session_state:
        if optimizer_defaults:
            st.session_state["backtest_tp_ratio"] = float(
                np.clip(
                    optimizer_defaults.get("atr_mult_tp", 2.0),
                    0.1,
                    15.0,
                )
            )
        else:
            st.session_state["backtest_tp_ratio"] = 2.0

    if optimizer_defaults:
        sma_fast_key = "sma_fast_SMA Crossover"
        if sma_fast_key not in st.session_state:
            st.session_state[sma_fast_key] = int(
                np.clip(
                    optimizer_defaults.get("sma_fast", 20),
                    HYBRID_BOUNDS["sma_fast"][0],
                    HYBRID_BOUNDS["sma_fast"][1],
                )
            )
        sma_slow_key = "sma_slow_SMA Crossover"
        if sma_slow_key not in st.session_state:
            st.session_state[sma_slow_key] = int(
                np.clip(
                    optimizer_defaults.get("sma_slow", 80),
                    HYBRID_BOUNDS["sma_slow"][0],
                    HYBRID_BOUNDS["sma_slow"][1],
                )
            )
        rsi_key = "rsi_period_RSI"
        if rsi_key not in st.session_state:
            st.session_state[rsi_key] = int(
                np.clip(
                    optimizer_defaults.get("rsi_period", 14),
                    HYBRID_BOUNDS["rsi_period"][0],
                    HYBRID_BOUNDS["rsi_period"][1],
                )
            )

    symbol = st.selectbox("Símbolo", ["BTCUSDT", "ETHUSDT"])
    interval = st.selectbox("Intervalo", ["1h", "4h", "1d"])

    capital_inicial = st.number_input(
        "💰 Capital inicial (USDT)", min_value=10.0, value=1000.0, step=10.0
    )
    riesgo_por_trade = st.number_input(
        "🎯 Riesgo por operación (%)", min_value=0.1, value=1.0, step=0.1
    )
    sl_ratio = st.number_input(
        "🛑 Stop Loss (R)",
        min_value=0.1,
        value=float(st.session_state.get("backtest_sl_ratio", 1.0)),
        step=0.1,
        key="backtest_sl_ratio",
    )
    tp_ratio = st.number_input(
        "🎯 Take Profit (R)",
        min_value=0.1,
        value=float(st.session_state.get("backtest_tp_ratio", 2.0)),
        step=0.1,
        key="backtest_tp_ratio",
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
                    result = run_visual_backtest(
                        df,
                        estrategias_seleccionadas,
                        strategy_params,
                        capital_inicial=capital_inicial,
                        riesgo_por_trade=riesgo_por_trade,
                        sl_ratio=sl_ratio,
                        tp_ratio=tp_ratio,
                        logica=logica_combinacion,
                        symbol=symbol,
                    )
                saved_run = save_backtest_run(
                    result,
                    symbol=symbol,
                    interval=interval,
                    strategies=estrategias_seleccionadas,
                    params=strategy_params,
                    capital_inicial=capital_inicial,
                    riesgo_por_trade=riesgo_por_trade,
                    sl_ratio=sl_ratio,
                    tp_ratio=tp_ratio,
                    logica=logica_combinacion,
                )
                st.session_state["backtest_result"] = {
                    "result": result,
                    "symbol": symbol,
                    "interval": interval,
                    "strategies": estrategias_seleccionadas,
                    "capital_inicial": capital_inicial,
                    "riesgo": riesgo_por_trade,
                    "sl_ratio": sl_ratio,
                    "tp_ratio": tp_ratio,
                    "logica": logica_combinacion,
                    "run_id": saved_run.run_id,
                    "metadata": saved_run.metadata,
                }
                st.success(
                    "✅ Backtest completado y guardado en la sección 📊 Analytics."
                )
            except ValueError as exc:
                st.error(f"❌ {exc}")

    if st.button("🚀 Optimizar y Backtestear"):
        if df is None or df.empty:
            st.warning("Debes descargar datos válidos antes de optimizar.")
        else:
            required_strategies = {"SMA Crossover", "RSI"}
            current_strategies = set(estrategias_seleccionadas)
            if not required_strategies.issubset(current_strategies):
                st.warning(
                    "La optimización avanzada requiere que estén activas las estrategias "
                    "SMA Crossover y RSI."
                )
            else:

                st.info("Iniciando optimización híbrida... esto puede tardar unos minutos ⏳")
                try:
                    hybrid_result = run_hybrid_optimization(
                        data=df,
                        strategies=estrategias_seleccionadas,
                        capital_inicial=capital_inicial,
                        riesgo_por_trade=riesgo_por_trade,
                        logica=logica_combinacion,
                        symbol=symbol,
                    )
                except Exception as exc:  # pragma: no cover - feedback para UI
                    st.error(f"No se pudo completar la optimización híbrida: {exc}")
                else:
                    st.success("✅ Optimización completada")

                    best_params = hybrid_result.best_params
                    optimizer_df = hybrid_result.history.copy()
                    if not optimizer_df.empty:
                        optimizer_df = optimizer_df.rename(columns={"expectancy_r": "expectancy_R"})
                        for column in [
                            "sma_fast",
                            "sma_slow",
                            "rsi_period",
                            "atr_mult_sl",
                            "atr_mult_tp",
                            "risk_per_trade",
                            "expectancy_R",
                            "profit_factor",
                            "max_drawdown",
                        ]:
                            if column in optimizer_df.columns:
                                optimizer_df[column] = pd.to_numeric(
                                    optimizer_df[column], errors="coerce"
                                )
                    st.session_state["advanced_optimizer_best_params"] = {
                        **best_params,
                        "method": hybrid_result.method,
                    }

                    st.markdown("### 💡 Parámetros óptimos actuales (auto-seleccionados)")
                    st.caption(
                        "El motor híbrido combina optimización bayesiana y genética y selecciona automáticamente la variante con mejor expectancy y profit factor."
                    )
                    best_display = pd.DataFrame([best_params])
                    best_display.insert(0, "optimizer", hybrid_result.method)
                    st.dataframe(best_display, use_container_width=True)

                    strategy_params = {
                        "SMA Crossover": {
                            "fast": int(best_params["sma_fast"]),
                            "slow": int(best_params["sma_slow"]),
                        },
                        "RSI": {"period": int(best_params["rsi_period"])},
                    }
                    backtest_result = run_backtest(
                        data=df,
                        strategies=estrategias_seleccionadas,
                        params=strategy_params,
                        capital_inicial=capital_inicial,
                        riesgo_por_trade=best_params["risk_per_trade"],
                        sl_ratio=best_params["atr_mult_sl"],
                        tp_ratio=best_params["atr_mult_tp"],
                        logica=logica_combinacion,
                        symbol=symbol,
                    )
                    st.success("✅ Backtest completado con parámetros óptimos")

                    st.markdown("### 📈 Backtesting Results")
                    st.caption(
                        "Visualiza el desempeño histórico de la estrategia con tus parámetros actuales o los optimizados automáticamente."
                    )
                    metrics_df = pd.DataFrame([backtest_result["metrics"]]).T.rename(columns={0: "Valor"})
                    st.dataframe(metrics_df, use_container_width=True)

                    equity_df = backtest_result["equity"].copy()
                    trades_df = backtest_result["trades"].copy()

                    st.markdown("### 🧠 Optimizations History")
                    st.caption("Aquí puedes ver cómo el bot fue aprendiendo y adaptando sus parámetros al mercado.")
                    if not optimizer_df.empty:
                        history_df = optimizer_df.reset_index(drop=True)
                        history_df["iteration"] = history_df.index + 1
                        expectancy_fig = px.line(
                            history_df,
                            x="iteration",
                            y="expectancy_R",
                            color="method" if "method" in history_df.columns else None,
                            title="Expectancy por iteración",
                        )
                        st.plotly_chart(expectancy_fig, use_container_width=True)
                        if "max_drawdown" in history_df.columns and history_df["max_drawdown"].notna().any():
                            drawdown_fig = px.line(
                                history_df,
                                x="iteration",
                                y="max_drawdown",
                                color="method" if "method" in history_df.columns else None,
                                title="Drawdown por iteración",
                            )
                            st.plotly_chart(drawdown_fig, use_container_width=True)
                        corr_cols = [
                            "sma_fast",
                            "sma_slow",
                            "rsi_period",
                            "atr_mult_sl",
                            "atr_mult_tp",
                            "risk_per_trade",
                            "expectancy_R",
                            "profit_factor",
                        ]
                        corr_df = history_df[corr_cols].dropna()
                        if not corr_df.empty:
                            st.markdown("### 🔥 Correlación entre parámetros y resultados")
                            heatmap = px.imshow(
                                corr_df.corr(),
                                text_auto=".2f",
                                color_continuous_scale="RdBu_r",
                                title="Heatmap de correlación parámetros vs resultados",
                            )
                            st.plotly_chart(heatmap, use_container_width=True)
                    else:
                        st.info("Aún no hay historial suficiente para graficar las iteraciones de optimización.")

                    st.markdown("### 📉 Equity Curve y Drawdown")
                    st.caption(
                        "Curva de patrimonio acumulado a lo largo del backtest. Muestra crecimiento y caídas temporales (drawdown)."
                    )
                    if not equity_df.empty:
                        equity_index = equity_df.columns[0]
                        equity_fig = px.line(equity_df, x=equity_index, y="equity", title="Curva de Equity")
                        st.plotly_chart(equity_fig, use_container_width=True)
                    raw_result = backtest_result.get("raw_result")
                    if raw_result is not None and getattr(raw_result, "drawdown", None) is not None:
                        drawdown_series = raw_result.drawdown * 100
                        drawdown_df = drawdown_series.reset_index()
                        drawdown_df.columns = ["timestamp", "drawdown_pct"]
                        drawdown_area = px.area(
                            drawdown_df,
                            x="timestamp",
                            y="drawdown_pct",
                            title="Drawdown %",
                        )
                        st.plotly_chart(drawdown_area, use_container_width=True)

                    if not trades_df.empty and "r_multiple" in trades_df.columns:
                        st.markdown("### 🎯 Distribución de R-Múltiplos")
                        st.caption("Indica cuántas operaciones alcanzaron distintos niveles de beneficio o pérdida.")
                        trades_df = trades_df.copy()
                        trades_df["trade_outcome"] = np.where(
                            trades_df["r_multiple"] > 0, "Ganadora", "Perdedora"
                        )
                        distribution_fig = px.histogram(
                            trades_df,
                            x="r_multiple",
                            nbins=25,
                            color="trade_outcome",
                            title="Distribución de R-Múltiplos",
                        )
                        st.plotly_chart(distribution_fig, use_container_width=True)

                    if not optimizer_df.empty:
                        st.markdown("### 🌐 3D Expectancy Map (ATR SL/TP vs RSI)")
                        st.caption(
                            "Mapa tridimensional que muestra cómo varía la rentabilidad esperada según los parámetros principales."
                        )
                        map_df = optimizer_df.dropna(subset=["atr_mult_sl", "atr_mult_tp", "expectancy_R"])
                        if not map_df.empty:
                            scatter_kwargs = {
                                "x": "atr_mult_sl",
                                "y": "atr_mult_tp",
                                "z": "expectancy_R",
                                "hover_data": [
                                    col
                                    for col in ["sma_fast", "sma_slow", "rsi_period", "method"]
                                    if col in map_df.columns
                                ],
                                "title": "Mapa 3D de Expectancy",
                            }
                            if "profit_factor" in map_df.columns:
                                scatter_kwargs["color"] = "profit_factor"
                            if "risk_per_trade" in map_df.columns:
                                scatter_kwargs["size"] = "risk_per_trade"
                            map_fig = px.scatter_3d(map_df, **scatter_kwargs)
                            st.plotly_chart(map_fig, use_container_width=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_dir = Path("results") / "runs" / timestamp
                    run_dir.mkdir(parents=True, exist_ok=True)

                    optimizer_path = Path("results/optimizer_results.csv")
                    optimizer_path.parent.mkdir(parents=True, exist_ok=True)
                    optimizer_df.to_csv(optimizer_path, index=False)

                    metrics_export = pd.DataFrame([backtest_result["metrics"]])
                    metrics_export.to_csv(Path("results/best_backtest.csv"), index=False)

                    optimizer_df.to_csv(run_dir / "optimizer_results.csv", index=False)
                    metrics_export.to_csv(run_dir / "best_backtest.csv", index=False)

                    equity_df.to_csv(Path("results/equity_curve.csv"), index=False)
                    trades_df.to_csv(Path("results/trades.csv"), index=False)
                    equity_df.to_csv(run_dir / "equity_curve.csv", index=False)
                    trades_df.to_csv(run_dir / "trades.csv", index=False)

                    optimizer_summary = {
                        "method": hybrid_result.method,
                        "bayesian": hybrid_result.bayesian.metrics,
                        "genetic": hybrid_result.genetic.metrics,
                    }
                    config_payload = {
                        "optimizer": {
                            "mode": "hibrido",
                            "strategies": list(estrategias_seleccionadas),
                            "best_params": best_params,
                            "summary": optimizer_summary,
                        },
                        "backtest": {
                            "capital_inicial": capital_inicial,
                            "riesgo_por_trade": best_params["risk_per_trade"],
                            "sl_ratio": best_params["atr_mult_sl"],
                            "tp_ratio": best_params["atr_mult_tp"],
                            "logica": logica_combinacion,
                            "symbol": symbol,
                        },
                    }
                    (run_dir / "config.json").write_text(
                        json.dumps(config_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                    pdf_path = generate_pdf_report(best_params, backtest_result, optimizer_df, timestamp)
                    st.success("📄 Reporte PDF generado exitosamente")

                    st.session_state["auto_optimizer_run"] = {
                        "metrics": backtest_result["metrics"],
                        "optimizer_results": optimizer_df.to_dict("records"),
                        "equity": equity_df.to_dict("records"),
                        "trades": trades_df.to_dict("records"),
                        "run_dir": str(run_dir),
                        "timestamp": timestamp,
                        "pdf": pdf_path,
                        "method": hybrid_result.method,
                    }


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

elif tab == "📊 Analytics":
    st.title("📊 Valvulin Backtest Analytics")

    st.sidebar.header("📁 Resultados del Backtest")
    saved_runs = list_backtest_runs()
    saved_lookup = {run.label(): run for run in saved_runs}
    manual_option = "📤 Cargar archivos manualmente"
    select_options = [manual_option, *saved_lookup.keys()]
    default_index = 0 if not saved_runs else 1
    selected_option = st.sidebar.selectbox(
        "Selecciona un backtest guardado",
        select_options,
        index=min(default_index, len(select_options) - 1),
    )

    trades_df: pd.DataFrame | None = None
    summary_df: pd.DataFrame | Mapping[str, float] | None = None
    metadata_selected: Dict[str, Any] | None = None
    translation_applied = False

    if selected_option != manual_option:
        selected_run = saved_lookup[selected_option]
        trades_raw, summary_raw = load_backtest_run(selected_run)
        trades_df = trades_raw
        if summary_raw is not None:
            if set(summary_raw.columns) >= {"metric", "value"}:
                summary_df = dict(zip(summary_raw["metric"], summary_raw["value"]))
            else:
                summary_df = summary_raw
        metadata_selected = selected_run.metadata
    else:
        trades_file = st.sidebar.file_uploader(
            "Carga el CSV de trades", type=["csv"], key="analytics_trades_uploader"
        )
        trades_path_input = st.sidebar.text_input(
            "Ruta alternativa del CSV de trades",
            value="",
            key="analytics_trades_path",
        )

        summary_file = st.sidebar.file_uploader(
            "Carga el CSV de métricas (opcional)",
            type=["csv"],
            key="analytics_summary_uploader",
        )
        summary_path_input = st.sidebar.text_input(
            "Ruta alternativa del CSV de métricas (opcional)",
            value="",
            key="analytics_summary_path",
        )

        if trades_file is not None:
            try:
                trades_df = pd.read_csv(trades_file)
            except Exception as exc:  # pragma: no cover - defensive UI feedback
                st.sidebar.error(f"No se pudo leer el CSV de trades: {exc}")
        elif trades_path_input:
            trades_path = Path(trades_path_input).expanduser()
            if trades_path.exists():
                try:
                    trades_df = pd.read_csv(trades_path)
                except Exception as exc:  # pragma: no cover - defensive UI feedback
                    st.sidebar.error(f"No se pudo leer el CSV de trades: {exc}")
            else:
                st.sidebar.warning("La ruta de trades especificada no existe.")

        if summary_file is not None:
            try:
                summary_df = pd.read_csv(summary_file)
            except Exception as exc:  # pragma: no cover - defensive UI feedback
                st.sidebar.error(f"No se pudo leer el CSV de métricas: {exc}")
        elif summary_path_input:
            summary_path = Path(summary_path_input).expanduser()
            if summary_path.exists():
                try:
                    summary_df = pd.read_csv(summary_path)
                except Exception as exc:  # pragma: no cover - defensive UI feedback
                    st.sidebar.error(f"No se pudo leer el CSV de métricas: {exc}")
            else:
                st.sidebar.warning("La ruta de métricas especificada no existe.")

    if trades_df is None:
        if saved_runs:
            st.info("Selecciona un backtest guardado o carga archivos manualmente para comenzar.")
        else:
            st.info("Carga un CSV de operaciones para visualizar los resultados del backtest.")
    else:
        trades_copy = trades_df.copy()
        trades_copy, translation_applied = normalize_trade_dataframe(trades_copy)

        required_columns = {
            "timestamp",
            "symbol",
            "side",
            "entry_price",
            "exit_price",
            "r_multiple",
            "capital_final",
        }
        missing_columns = required_columns.difference(trades_copy.columns)
        if missing_columns:
            st.error(
                "El CSV de trades debe contener las columnas requeridas: "
                + ", ".join(sorted(missing_columns))
            )
        else:
            if translation_applied:
                st.sidebar.info(
                    "✅ CSV detectado correctamente. Columnas normalizadas para análisis."
                )

            trades_copy["timestamp"] = pd.to_datetime(
                trades_copy["timestamp"], errors="coerce"
            )
            for numeric_column in [
                "entry_price",
                "exit_price",
                "r_multiple",
                "capital_final",
            ]:
                trades_copy[numeric_column] = pd.to_numeric(
                    trades_copy[numeric_column], errors="coerce"
                )
            trades_copy.sort_values(by="timestamp", inplace=True)
            trades_copy.dropna(subset=["capital_final", "r_multiple"], inplace=True)

            if metadata_selected:
                capital_inicial_meta = metadata_selected.get("capital_inicial")
                capital_final_meta = metadata_selected.get("capital_final")
                capital_inicial_value = (
                    float(capital_inicial_meta)
                    if capital_inicial_meta is not None
                    else 0.0
                )
                capital_final_value = (
                    float(capital_final_meta)
                    if capital_final_meta is not None
                    else 0.0
                )
                strategy_label = metadata_selected.get("strategy_label", "-")
                symbol_label = metadata_selected.get("symbol", "-")
                executed_at = metadata_selected.get("executed_at")
                executed_display = executed_at
                if isinstance(executed_at, str):
                    try:
                        executed_display = datetime.fromisoformat(executed_at).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                    except ValueError:
                        executed_display = executed_at
                info_box = f"""
                <div style="background-color:#f5f7ff;border-radius:10px;padding:16px;margin-bottom:1rem;">
                    <strong>Estrategia:</strong> {strategy_label}<br/>
                    <strong>Símbolo:</strong> {symbol_label}<br/>
                    <strong>Capital inicial:</strong> ${capital_inicial_value:,.2f} &nbsp;→&nbsp;
                    <strong>Capital final:</strong> ${capital_final_value:,.2f}<br/>
                    <small>Ejecutado: {executed_display}</small>
                </div>
                """
                st.markdown(info_box, unsafe_allow_html=True)

                st.sidebar.markdown("### 📐 Parámetros de la estrategia")
                parameters = metadata_selected.get("parameters") or {}
                if parameters:
                    for name, value in parameters.items():
                        if isinstance(value, dict):
                            st.sidebar.markdown(f"**{name}**")
                            for sub_key, sub_value in value.items():
                                st.sidebar.markdown(f"- {sub_key}: {sub_value}")
                        else:
                            st.sidebar.markdown(f"- **{name}**: {value}")
                else:
                    st.sidebar.info("No hay parámetros personalizados registrados para este backtest.")

            st.sidebar.subheader("🔍 Filtros")
            strategy_col = "strategy" if "strategy" in trades_copy.columns else None
            if strategy_col:
                strategy_options = ["Todas"] + sorted(
                    trades_copy[strategy_col].dropna().astype(str).unique().tolist()
                )
            else:
                strategy_options = ["Todas"]
            selected_strategy = st.sidebar.selectbox(
                "Selecciona estrategia", strategy_options
            )

            symbol_options = ["Todos"] + sorted(
                trades_copy["symbol"].dropna().astype(str).unique().tolist()
            )
            selected_symbol = st.sidebar.selectbox("Selecciona símbolo", symbol_options)

            date_range_value = None
            if trades_copy["timestamp"].notna().any():
                valid_timestamps = trades_copy["timestamp"].dropna()
                min_date = valid_timestamps.min().date()
                max_date = valid_timestamps.max().date()
                date_range_value = st.sidebar.date_input(
                    "Rango de fechas",
                    value=(min_date, max_date),
                )

            filtered_df = trades_copy
            if strategy_col and selected_strategy != "Todas":
                filtered_df = filtered_df[
                    filtered_df[strategy_col].astype(str) == selected_strategy
                ]
            if selected_symbol != "Todos":
                filtered_df = filtered_df[filtered_df["symbol"] == selected_symbol]

            if date_range_value:
                if isinstance(date_range_value, tuple):
                    start_date, end_date = date_range_value
                elif isinstance(date_range_value, list) and len(date_range_value) == 2:
                    start_date, end_date = date_range_value
                else:
                    start_date = end_date = date_range_value

                if start_date and end_date:
                    if start_date > end_date:
                        start_date, end_date = end_date, start_date
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(
                        microseconds=1
                    )

                    timestamp_series = filtered_df["timestamp"]
                    if pd.api.types.is_datetime64_any_dtype(timestamp_series):
                        tz = None
                        if pd.api.types.is_datetime64tz_dtype(timestamp_series):
                            tz = timestamp_series.dt.tz

                        if tz is not None:
                            if start_ts.tzinfo is None:
                                start_ts = start_ts.tz_localize(tz)
                            else:
                                start_ts = start_ts.tz_convert(tz)

                            if end_ts.tzinfo is None:
                                end_ts = end_ts.tz_localize(tz)
                            else:
                                end_ts = end_ts.tz_convert(tz)

                        filtered_df = filtered_df[timestamp_series.between(start_ts, end_ts)]
                    else:
                        timestamp_values = pd.to_datetime(timestamp_series, errors="coerce")
                        filtered_df = filtered_df[
                            timestamp_values.between(start_ts, end_ts)
                        ]

            if filtered_df.empty:
                st.warning("No hay operaciones para los filtros seleccionados.")
            else:
                enriched_df = prepare_drawdown_columns(filtered_df)
                summary_payload = summary_df
                if isinstance(summary_payload, pd.DataFrame) and set(summary_payload.columns) >= {
                    "metric",
                    "value",
                }:
                    summary_payload = dict(
                        zip(summary_payload["metric"], summary_payload["value"])
                    )
                summary_metrics = compute_backtest_summary_metrics(
                    filtered_df,
                    summary_payload,
                )

                st.subheader("🧾 Summary Metrics")
                metric_labels = [
                    "Winrate %",
                    "Net Profit %",
                    "Expectancy (R)",
                    "Average Win (R)",
                    "Average Loss (R)",
                    "RR Effective",
                    "Breakeven Winrate %",
                    "Max Drawdown %",
                    "Profit Factor",
                ]
                metric_columns = st.columns(3)
                for idx, label in enumerate(metric_labels):
                    column = metric_columns[idx % 3]
                    value = summary_metrics.get(label, 0.0)
                    if label.endswith("%"):
                        column.metric(label, f"{value:.2f}%")
                    else:
                        column.metric(label, f"{value:.2f}")

                st.subheader("📈 Equity Curve")
                equity_fig = plot_equity_curve(
                    enriched_df, drawdown=enriched_df["drawdown"]
                )
                st.plotly_chart(equity_fig, use_container_width=True)

                st.subheader("📉 Drawdown Curve")
                drawdown_fig = plot_drawdown_curve(enriched_df)
                st.plotly_chart(drawdown_fig, use_container_width=True)

                st.subheader("📊 R-Multiple Distribution")
                r_multiple_fig = plot_r_multiple_distribution(filtered_df)
                st.plotly_chart(r_multiple_fig, use_container_width=True)

                st.subheader("📈 Expectancy")
                expectancy_fig = plot_expectancy_bar(
                    summary_metrics.get("Expectancy (R)", 0.0)
                )
                st.plotly_chart(expectancy_fig, use_container_width=False)

                with st.expander("Ver operaciones filtradas"):
                    preview_df = filtered_df.copy()
                    preview_df["timestamp"] = preview_df["timestamp"].dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    st.dataframe(preview_df, use_container_width=True)

                compare_labels = st.sidebar.multiselect(
                    "Comparar hasta dos backtests",
                    list(saved_lookup.keys()),
                    help="Selecciona dos backtests guardados para comparar sus métricas.",
                )
                if len(compare_labels) > 2:
                    st.sidebar.warning("Selecciona máximo dos backtests para el comparador.")
                    compare_labels = compare_labels[:2]

                if len(compare_labels) == 2:
                    st.subheader("⚖️ Comparador de estrategias")

                    comparison_rows = []
                    for label in compare_labels:
                        run_summary = saved_lookup.get(label)
                        if not run_summary:
                            continue
                        trades_run, metrics_run = load_backtest_run(run_summary)
                        trades_norm, _ = normalize_trade_dataframe(trades_run.copy())
                        trades_norm["timestamp"] = pd.to_datetime(
                            trades_norm.get("timestamp"), errors="coerce"
                        )
                        trades_norm = trades_norm.dropna(
                            subset=["timestamp", "capital_final", "r_multiple"]
                        )
                        metrics_mapping: Mapping[str, float] | None = None
                        if metrics_run is not None:
                            if set(metrics_run.columns) >= {"metric", "value"}:
                                metrics_mapping = dict(
                                    zip(metrics_run["metric"], metrics_run["value"])
                                )
                            else:
                                metrics_mapping = metrics_run.iloc[0].to_dict()

                        metrics_values = compute_backtest_summary_metrics(
                            trades_norm,
                            metrics_mapping,
                        )
                        metadata = run_summary.metadata
                        comparison_rows.append(
                            {
                                "Backtest": label,
                                "Estrategia": metadata.get("strategy_label", "-"),
                                "Símbolo": metadata.get("symbol", "-"),
                                "Capital Inicial": metadata.get("capital_inicial", 0.0),
                                "Capital Final": metadata.get("capital_final", 0.0),
                                "Winrate %": metrics_values.get("Winrate %", 0.0),
                                "Expectancy (R)": metrics_values.get("Expectancy (R)", 0.0),
                                "Profit Factor": metrics_values.get("Profit Factor", 0.0),
                                "Max Drawdown %": metrics_values.get("Max Drawdown %", 0.0),
                            }
                        )

                    if comparison_rows:
                        comparison_df = pd.DataFrame(comparison_rows)
                        st.dataframe(
                            comparison_df.set_index("Backtest"), use_container_width=True
                        )

    st.divider()
    optim_tab, = st.tabs(["🧠 Optimización Avanzada"])
    with optim_tab:
        st.subheader("🧠 Optimización Avanzada")
        st.markdown(
            "Consulta los resultados más recientes de la optimización automática y "
            "analiza las métricas, curvas de equity y distribuciones de R-múltiplos."
        )

        optimizer_path = Path("results/optimizer_results.csv")
        backtest_path = Path("results/best_backtest.csv")
        equity_path = Path("results/equity_curve.csv")
        trades_path = Path("results/trades.csv")

        if not optimizer_path.exists() or optimizer_path.stat().st_size == 0:
            st.info(
                "Ejecuta el flujo de optimización automática en la pestaña 🔁 Backtesting "
                "para generar resultados."
            )
        else:
            try:
                optimizer_df = pd.read_csv(optimizer_path)
            except Exception as exc:
                st.error(f"No se pudo leer `optimizer_results.csv`: {exc}")
                optimizer_df = pd.DataFrame()

            if optimizer_df.empty:
                st.info("Aún no hay resultados de optimización para mostrar.")
            else:
                numeric_cols = [
                    "ema_fast",
                    "ema_slow",
                    "rsi_period",
                    "atr_mult_sl",
                    "atr_mult_tp",
                    "expectancy_R",
                    "max_drawdown",
                    "profit_factor",
                    "rr_effective",
                    "fitness",
                ]
                for column in numeric_cols:
                    if column in optimizer_df.columns:
                        optimizer_df[column] = pd.to_numeric(
                            optimizer_df[column], errors="coerce"
                        )
                optimizer_df.dropna(
                    subset=["expectancy_R", "max_drawdown"], inplace=True
                )
                optimizer_df.sort_values(
                    by=["expectancy_R", "profit_factor", "max_drawdown"],
                    ascending=[False, False, True],
                    inplace=True,
                )

                st.write("### 🏆 Top 10 configuraciones optimizadas")
                st.dataframe(
                    optimizer_df.head(10).reset_index(drop=True),
                    use_container_width=True,
                )

                if {"atr_mult_sl", "atr_mult_tp", "expectancy_R"}.issubset(
                    optimizer_df.columns
                ):
                    scatter_df = optimizer_df.head(200)
                    scatter_fig = px.scatter_3d(
                        scatter_df,
                        x="atr_mult_sl",
                        y="atr_mult_tp",
                        z="expectancy_R",
                        color="max_drawdown",
                        size="profit_factor",
                        hover_data=["ema_fast", "ema_slow", "rsi_period"],
                        title="Mapa 3D Expectancy vs ATR",
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)

            if backtest_path.exists() and backtest_path.stat().st_size > 0:
                try:
                    metrics_df = pd.read_csv(backtest_path)
                except Exception as exc:
                    st.error(f"No se pudo leer `best_backtest.csv`: {exc}")
                    metrics_df = pd.DataFrame()
                if not metrics_df.empty:
                    st.write("### 📊 Métricas del mejor backtest")
                    st.dataframe(metrics_df.T.rename(columns={0: "Valor"}))

            equity_df = None
            if equity_path.exists() and equity_path.stat().st_size > 0:
                try:
                    equity_df = pd.read_csv(equity_path)
                except Exception as exc:
                    st.warning(f"No se pudo leer la curva de equity: {exc}")
                    equity_df = None

            if equity_df is None and "auto_optimizer_run" in st.session_state:
                stored_equity = st.session_state["auto_optimizer_run"].get("equity")
                if stored_equity:
                    equity_df = pd.DataFrame(stored_equity)

            if equity_df is not None and not equity_df.empty:
                equity_fig = px.line(
                    equity_df,
                    x=equity_df.columns[0],
                    y="equity",
                    title="Curva de equity (último run)",
                )
                st.plotly_chart(equity_fig, use_container_width=True)

            trades_df = None
            if trades_path.exists() and trades_path.stat().st_size > 0:
                try:
                    trades_df = pd.read_csv(trades_path)
                except Exception as exc:
                    st.warning(f"No se pudo leer el historial de trades: {exc}")
                    trades_df = None

            if trades_df is None and "auto_optimizer_run" in st.session_state:
                stored_trades = st.session_state["auto_optimizer_run"].get("trades")
                if stored_trades:
                    trades_df = pd.DataFrame(stored_trades)

            if trades_df is not None and not trades_df.empty and "r_multiple" in trades_df.columns:
                trades_df["r_multiple"] = pd.to_numeric(
                    trades_df["r_multiple"], errors="coerce"
                )
                trades_df.dropna(subset=["r_multiple"], inplace=True)
                if not trades_df.empty:
                    hist_fig = px.histogram(
                        trades_df,
                        x="r_multiple",
                        nbins=25,
                        title="Distribución de R-múltiplos",
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)

elif tab == "⚙️ Configuración":
    st.title("⚙️ Configuración y ayuda")
    st.write("### ⚙️ Configuración y ayuda")
    st.write(
        "Utiliza esta sección para gestionar archivos de configuración avanzados, documentar tus estrategias o añadir notas de operación."
    )
    st.write(
        "- Descarga datos desde la pestaña **Datos**.\n"
        "- Ejecuta backtests visuales en la pestaña **Backtesting**.\n"
        "- Personaliza tus `config.yaml` con los botones disponibles tras cada descarga."
    )
