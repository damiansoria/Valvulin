"""DataFeed público para descargar datos históricos desde Binance."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import logging
import time
from typing import Callable, Iterable, List, Optional

import pandas as pd
import requests

from .feeds import MarketDataRequest


logger = logging.getLogger(__name__)


RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_LIMIT = 1000


def _to_iso_utc(timestamp_ms: int) -> str:
    """Convertir un timestamp en milisegundos a una cadena ISO UTC."""

    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


@dataclass
class _ChunkInfo:
    """Información auxiliar para el progreso de descarga."""

    count: int
    last_open_time_ms: int


class BinancePublicDataFeed:
    """Descarga datos OHLCV desde el endpoint público de Binance."""

    def __init__(
        self,
        data_dir: str | Path = "data/history",
        base_url: str = "https://api.binance.com",
        rate_sleep: float = 0.35,
        compress: bool = False,
        cache: bool = True,
        max_limit: int = MAX_LIMIT,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.base_url = base_url.rstrip("/")
        self.rate_sleep = rate_sleep
        self.compress = compress
        self.cache = cache
        self._session = requests.Session()
        self.max_limit = max(1, min(max_limit, MAX_LIMIT))

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        chunk_callback: Optional[Callable[["_ChunkInfo", int], None]] = None,
    ) -> pd.DataFrame:
        """Descargar velas históricas usando la API pública de Binance."""

        url = f"{self.base_url}/api/v3/klines"
        limit = max(1, min(self.max_limit, MAX_LIMIT))
        params_base = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        logger.info(
            "binance_fetch_start",
            extra={"symbol": symbol, "interval": interval, "start_time": start_time, "end_time": end_time},
        )

        rows: List[dict] = []
        next_start = start_time
        previous_start = None

        while True:
            params = params_base.copy()
            if next_start is not None:
                params["startTime"] = next_start
            if end_time is not None:
                params["endTime"] = end_time

            data = self._request(url, params)

            if not data:
                break

            chunk_info = self._process_chunk(data, rows)
            logger.info(
                "binance_fetch_chunk",
                extra={
                    "symbol": symbol,
                    "interval": interval,
                    "rows": chunk_info.count,
                    "last_open_time": _to_iso_utc(chunk_info.last_open_time_ms),
                },
            )

            if chunk_callback is not None:
                try:
                    chunk_callback(chunk_info, len(rows))
                except Exception as exc:  # pragma: no cover - feedback externo
                    logger.warning("binance_progress_callback_error", extra={"error": str(exc)})

            if len(data) < limit:
                break

            last_open_time = chunk_info.last_open_time_ms
            next_start = last_open_time + 1

            if end_time is not None and next_start > end_time:
                break

            if previous_start is not None and next_start <= previous_start:
                # Protege contra bucles infinitos si la API devuelve datos repetidos.
                break

            previous_start = next_start

            if self.rate_sleep:
                time.sleep(self.rate_sleep)

        if not rows:
            return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

        frame = pd.DataFrame(rows)
        frame.drop_duplicates(subset="open_time", inplace=True)
        frame.sort_values("open_time", inplace=True)
        frame.reset_index(drop=True, inplace=True)

        logger.info(
            "binance_fetch_complete",
            extra={"symbol": symbol, "interval": interval, "total_rows": len(frame)},
        )
        return frame

    def download_to_csv(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        out_path: Optional[str | Path] = None,
        compress: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, Optional[str]], None]] = None,
        chunk_callback: Optional[Callable[["_ChunkInfo", int], None]] = None,
    ) -> Path:
        """Descargar datos y almacenarlos en CSV (con opción de compresión).

        El callback ``progress_callback`` recibe el porcentaje estimado (0-100)
        y un mensaje descriptivo. ``chunk_callback`` mantiene compatibilidad con
        la retroalimentación detallada de cada bloque descargado.
        """

        use_compress = self.compress if compress is None else compress
        if out_path is None:
            extension = ".csv.gz" if use_compress else ".csv"
            out_path = self.data_dir / f"{symbol}_{interval}{extension}"
        else:
            out_path = Path(out_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing = self._load_existing(out_path) if self.cache else None
        fetch_start = start_time

        if existing is not None and not existing.empty:
            existing = self._normalize_frame(existing)
            last_open_time = existing["open_time"].max()
            if pd.isna(last_open_time):
                fetch_start = start_time
            else:
                last_timestamp = self._to_timestamp_ms(last_open_time)
                fetch_start = max(start_time or (last_timestamp + 1), last_timestamp + 1)

        if progress_callback is not None:
            try:
                progress_callback(0, "Preparando descarga...")
            except Exception as exc:  # pragma: no cover - feedback externo
                logger.warning("binance_progress_callback_error", extra={"error": str(exc)})

        reference_start = fetch_start
        target_end = end_time
        if reference_start is not None and target_end is None:
            target_end = int(datetime.now(timezone.utc).timestamp() * 1000)

        total_span_ms: Optional[int] = None
        if reference_start is not None and target_end is not None and target_end > reference_start:
            total_span_ms = target_end - reference_start

        last_pct_reported = 0

        def _forward_chunk(chunk: _ChunkInfo, total_rows: int) -> None:
            nonlocal last_pct_reported

            if chunk_callback is not None:
                try:
                    chunk_callback(chunk, total_rows)
                except Exception as exc:  # pragma: no cover - feedback externo
                    logger.warning("binance_progress_callback_error", extra={"error": str(exc)})

            if progress_callback is None:
                return

            pct: int
            message = f"Descargadas {total_rows} velas..."

            if total_span_ms is not None and reference_start is not None:
                covered = max(0, chunk.last_open_time_ms - reference_start)
                if covered >= total_span_ms:
                    pct = 99
                else:
                    pct = max(last_pct_reported, min(99, int((covered / total_span_ms) * 100)))
            else:
                pct = min(95, last_pct_reported + 5)

            if pct > last_pct_reported:
                last_pct_reported = pct

            try:
                progress_callback(last_pct_reported, message)
            except Exception as exc:  # pragma: no cover - feedback externo
                logger.warning("binance_progress_callback_error", extra={"error": str(exc)})

        internal_callback = None
        if progress_callback is not None or chunk_callback is not None:
            internal_callback = _forward_chunk

        fetched = self.fetch_klines(
            symbol,
            interval,
            start_time=fetch_start,
            end_time=end_time,
            chunk_callback=internal_callback,
        )

        if existing is not None and not existing.empty:
            combined = pd.concat([existing, fetched], ignore_index=True)
        else:
            combined = fetched

        combined = self._normalize_frame(combined)

        if use_compress:
            combined.to_csv(out_path, index=False, compression="gzip")
        else:
            combined.to_csv(out_path, index=False)

        if progress_callback is not None:
            try:
                progress_callback(100, "✅ Descarga completada")
            except Exception as exc:  # pragma: no cover - feedback externo
                logger.warning("binance_progress_callback_error", extra={"error": str(exc)})

        logger.info(
            "binance_download_complete",
            extra={"symbol": symbol, "interval": interval, "file": str(out_path), "rows": len(combined)},
        )
        return out_path

    def load(self, request: MarketDataRequest) -> pd.DataFrame:
        """Cargar datos desde CSV existente o descargarlos si no existen."""

        file_path = self._find_existing_file(request.symbol, request.interval)
        if file_path is None:
            start_ms = int(request.start.timestamp() * 1000) if request.start else None
            end_ms = int(request.end.timestamp() * 1000) if request.end else None
            file_path = self.download_to_csv(
                request.symbol,
                request.interval,
                start_time=start_ms,
                end_time=end_ms,
            )

        logger.info(
            "binance_load_csv",
            extra={"symbol": request.symbol, "interval": request.interval, "file": str(file_path)},
        )

        frame = pd.read_csv(file_path, parse_dates=["open_time"], infer_datetime_format=True)
        frame.sort_values("open_time", inplace=True)
        frame.drop_duplicates(subset="open_time", inplace=True)

        if request.start:
            frame = frame[frame["open_time"] >= request.start]
        if request.end:
            frame = frame[frame["open_time"] <= request.end]

        frame.reset_index(drop=True, inplace=True)
        return frame

    def _process_chunk(self, data: Iterable[list], rows: List[dict]) -> _ChunkInfo:
        for entry in data:
            open_time_ms = int(entry[0])
            rows.append(
                {
                    "open_time": _to_iso_utc(open_time_ms),
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                }
            )
        return _ChunkInfo(count=len(data), last_open_time_ms=int(data[-1][0]))

    def _request(self, url: str, params: dict) -> list:
        attempts = 0
        wait_time = 1.0
        while True:
            try:
                response = self._session.get(url, params=params, timeout=10)
            except requests.RequestException as exc:  # pragma: no cover - red externa
                logger.warning("binance_request_exception", extra={"error": str(exc)})
                if attempts >= 3:
                    raise RuntimeError("Error de conexión con Binance") from exc
                time.sleep(wait_time)
                wait_time *= 2
                attempts += 1
                continue

            if response.status_code == 200:
                return response.json()

            if response.status_code in RETRY_STATUS and attempts < 3:
                logger.warning(
                    "binance_request_retry",
                    extra={"status": response.status_code, "wait": wait_time, "params": params},
                )
                time.sleep(wait_time)
                wait_time *= 2
                attempts += 1
                continue

            message = response.text or "Respuesta inesperada de Binance"
            logger.error(
                "binance_request_failed",
                extra={"status": response.status_code, "message": message, "params": params},
            )
            raise RuntimeError(f"Error {response.status_code} al solicitar datos de Binance: {message}")

    def _load_existing(self, path: Path | None) -> Optional[pd.DataFrame]:
        if path is None or not path.exists():
            return None
        return pd.read_csv(path)

    def _find_existing_file(self, symbol: str, interval: str) -> Optional[Path]:
        plain = self.data_dir / f"{symbol}_{interval}.csv"
        gz = self.data_dir / f"{symbol}_{interval}.csv.gz"
        if plain.exists():
            return plain
        if gz.exists():
            return gz
        return None

    def _normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "open_time" in frame.columns:
            open_dt = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
            frame = frame.assign(open_time=open_dt)
            frame.dropna(subset=["open_time"], inplace=True)
            iso_series = frame["open_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            frame["open_time"] = iso_series.str.replace("+0000", "+00:00", regex=False)

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for column in numeric_cols:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame[["open_time", "open", "high", "low", "close", "volume"]].copy()
        frame.dropna(subset=["open_time"], inplace=True)
        frame.drop_duplicates(subset="open_time", inplace=True)
        frame.sort_values("open_time", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame

    def _to_timestamp_ms(self, value: object) -> int:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        return int(ts.timestamp() * 1000)


__all__ = ["BinancePublicDataFeed"]

