"""Interfaz Streamlit para descargar datos históricos públicos desde Binance."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import streamlit as st
import yaml

from valvulin.data.binance_public import BinancePublicDataFeed

# Configuración inicial de la página
st.set_page_config(page_title="Valvulin Trading Bot", layout="wide")

# Inicializar el estado para los registros de log y descargas recientes
if "logs" not in st.session_state:
    st.session_state["logs"] = []

if "pending_feeds" not in st.session_state:
    st.session_state["pending_feeds"] = {}

# Barra lateral de configuración básica
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

# Instancia del feed con los parámetros seleccionados
feed = BinancePublicDataFeed(rate_sleep=rate_sleep, compress=compress, cache=continue_download, max_limit=int(limit_value))

st.write("### Descargar o actualizar datos históricos")
st.info("Puedes usar esta herramienta para descargar datos públicos de Binance sin necesidad de API Key.")

# Mostrar tabla de feeds existentes si hay un config.yaml
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

        st.session_state["pending_feeds"] = {
            symbol: path for symbol, path in successful_downloads.items()
        }

        log_placeholder.text_area(
            "Registro de descargas",
            value="\n".join(st.session_state["logs"]),
            height=220,
            disabled=True,
        )

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

# Mostrar información de errores si quedaron pendientes
if errors:
    for error in errors:
        st.error(error)
