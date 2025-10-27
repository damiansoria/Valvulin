"""Script utilitario para probar descargas del feed público de Binance."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).resolve().parents[1]))

from valvulin.data.binance_public import BinancePublicDataFeed  # noqa: E402  # isort:skip


def _parse_start(value: str) -> int:
    """Convertir una fecha en formato YYYY-MM-DD a timestamp en milisegundos."""

    try:
        date_obj = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:  # pragma: no cover - entrada de usuario
        raise argparse.ArgumentTypeError("Usa el formato YYYY-MM-DD") from exc
    return int(datetime.combine(date_obj, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)


def _build_progress(symbol: str) -> Callable:
    """Crear un callback de progreso que imprime actualizaciones en consola."""

    def _progress(chunk, total_rows):
        timestamp = datetime.fromtimestamp(chunk.last_open_time_ms / 1000, tz=timezone.utc)
        print(
            f"{symbol}: bloque de {chunk.count} velas. Último timestamp {timestamp.isoformat()} (total={total_rows})",
            flush=True,
        )

    return _progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Descargar datos históricos públicos desde Binance.")
    parser.add_argument("symbol", help="Símbolo, por ejemplo BTCUSDT")
    parser.add_argument("interval", help="Intervalo, por ejemplo 1h")
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        type=_parse_start,
        help="Fecha inicial en formato YYYY-MM-DD (UTC).",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Número máximo de velas por request (1-1000).")
    parser.add_argument(
        "--rate-sleep",
        type=float,
        default=0.35,
        help="Tiempo de espera entre requests en segundos.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Guardar el archivo comprimido con extensión .csv.gz",
    )
    parser.add_argument(
        "--no-continue",
        action="store_true",
        help="No continuar la descarga desde el último punto disponible.",
    )

    args = parser.parse_args()

    symbol = args.symbol.upper()

    feed = BinancePublicDataFeed(
        rate_sleep=args.rate_sleep,
        compress=args.compress,
        cache=not args.no_continue,
        max_limit=max(1, min(args.limit, 1000)),
    )

    extension = ".csv.gz" if args.compress else ".csv"
    output_path = Path(f"data/history/{symbol}_{args.interval}{extension}")

    print(f"Descargando datos para {symbol} ({args.interval}) hacia {output_path}...")
    output = feed.download_to_csv(
        symbol,
        args.interval,
        start_time=args.start_date,
        out_path=output_path,
        compress=args.compress,
        progress_callback=_build_progress(symbol),
    )

    if output.exists():
        size_mb = round(output.stat().st_size / 1024 / 1024, 2)
        print(f"Archivo guardado correctamente en {output} ({size_mb} MB)")
    else:  # pragma: no cover - validación de seguridad
        raise SystemExit("La descarga no generó el archivo esperado")


if __name__ == "__main__":
    main()
