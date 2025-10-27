"""Utilidad para descargar datos históricos públicos desde Binance."""
from __future__ import annotations

from datetime import datetime, timezone

from valvulin.data.binance_public import BinancePublicDataFeed


if __name__ == "__main__":
    feed = BinancePublicDataFeed()
    start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    feed.download_to_csv("BTCUSDT", "1h", start_time=start)
    print("✅ Datos descargados y guardados en data/history/BTCUSDT_1h.csv")
