"""Simple CLI dashboard displaying the bot status."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from .performance import compute_performance_metrics, group_metrics_by_strategy
from .trade_logger import TradeLogger


def _format_table(rows: List[List[str]], headers: List[str]) -> str:
    col_widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_row(values: Iterable[str]) -> str:
        return " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(values))

    header_line = fmt_row(headers)
    separator = "-+-".join("-" * width for width in col_widths)
    body = "\n".join(fmt_row(row) for row in rows) if rows else "(sin datos)"
    return f"{header_line}\n{separator}\n{body}"


def load_dashboard(trade_logger: TradeLogger, recent: int = 5) -> str:
    trades = trade_logger.load_trades()
    open_trades = trade_logger.load_open_trades()

    metrics = compute_performance_metrics(trades)
    strategy_metrics = group_metrics_by_strategy(trades)

    lines = []
    lines.append("=== ESTADO GENERAL DEL BOT ===")
    lines.append(f"Total trades: {metrics.total_trades}")
    lines.append(f"Win rate: {metrics.win_rate:.2%}")
    lines.append(f"Profit factor: {metrics.profit_factor:.2f}")
    lines.append(f"Expectancy: {metrics.expectancy:.2f} R")
    lines.append(f"Max drawdown: {metrics.max_drawdown:.2f} R")
    lines.append(f"R acumulado: {metrics.total_r_multiple:.2f} R")
    lines.append("")

    if open_trades:
        rows = [
            [
                trade.symbol,
                trade.strategy,
                trade.opened_at.strftime("%Y-%m-%d %H:%M"),
                f"{trade.entry_price:.2f}",
                f"{trade.size:.2f}",
                f"{trade.stop_loss:.2f}",
                f"{trade.take_profit:.2f}",
            ]
            for trade in open_trades
        ]
        lines.append("=== OPERACIONES ABIERTAS ===")
        lines.append(
            _format_table(
                rows,
                [
                    "Símbolo",
                    "Estrategia",
                    "Abierta",
                    "Entrada",
                    "Tamaño",
                    "SL",
                    "TP",
                ],
            )
        )
        lines.append("")
    else:
        lines.append("No hay operaciones abiertas registradas.")
        lines.append("")

    if trades:
        recent_trades = sorted(trades, key=lambda t: t.timestamp, reverse=True)[:recent]
        rows = [
            [
                trade.timestamp.strftime("%Y-%m-%d %H:%M"),
                trade.strategy,
                f"{trade.r_multiple:.2f}",
                f"{trade.stop_loss:.2f}",
                f"{trade.take_profit:.2f}",
                trade.notes or "-",
            ]
            for trade in recent_trades
        ]
        lines.append("=== TRADES RECIENTES ===")
        lines.append(
            _format_table(
                rows,
                ["Fecha", "Estrategia", "R", "SL", "TP", "Notas"],
            )
        )
        lines.append("")

    if strategy_metrics:
        rows = [
            [
                name,
                f"{metrics.expectancy:.2f}",
                f"{metrics.win_rate:.1%}",
                f"{metrics.total_trades}",
            ]
            for name, metrics in strategy_metrics.items()
        ]
        lines.append("=== MÉTRICAS POR ESTRATEGIA ===")
        lines.append(
            _format_table(
                rows,
                ["Estrategia", "Expectancy", "Win Rate", "# Trades"],
            )
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard CLI del bot")
    parser.add_argument(
        "--trades-path",
        type=Path,
        default=Path("data/trades.csv"),
        help="Ruta al CSV de trades cerrados",
    )
    parser.add_argument(
        "--open-trades-path",
        type=Path,
        default=Path("data/open_trades.csv"),
        help="Ruta al CSV de operaciones abiertas",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=5,
        help="Número de trades recientes a mostrar",
    )
    args = parser.parse_args()

    logger = TradeLogger(trades_path=args.trades_path, open_trades_path=args.open_trades_path)
    dashboard_text = load_dashboard(logger, recent=args.recent)
    print(dashboard_text)


if __name__ == "__main__":
    main()
