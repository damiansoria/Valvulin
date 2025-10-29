"""Historical backtesting utilities.

The module provides a feature rich :class:`Backtester` capable of simulating
multiple trading strategies on OHLCV data downloaded from Binance.  The
implementation focuses on risk-first behaviour: each trade is sized using the
current equity, commissions are applied, drawdown protections are enforced and a
complete equity curve is produced.  The helper classes are intentionally
independent from any exchange specific code so that they can be reused in unit
tests and analytics notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analytics.metrics import trade_distribution_metrics

from .types import OrderStatus, TradeSignal

if TYPE_CHECKING:
    from analytics.trade_logger import TradeLogger


DEFAULT_INITIAL_EQUITY = 1_000.0
DEFAULT_ATR_PERIOD = 14


def _validate_percentage(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be greater or equal than zero")
    return float(value)


@dataclass(slots=True)
class BacktestSettings:
    """Container for the configurable aspects of a backtest run."""

    risk_per_trade_pct: float = 1.0
    rr_ratio: float = 2.0
    sl_ratio: float = 1.0
    commission_pct: float = 0.1

    def __post_init__(self) -> None:
        _validate_percentage("risk_per_trade_pct", self.risk_per_trade_pct)
        if self.rr_ratio <= 0:
            raise ValueError("rr_ratio must be positive")
        if self.sl_ratio <= 0:
            raise ValueError("sl_ratio must be positive")
        _validate_percentage("commission_pct", self.commission_pct)


@dataclass(slots=True)
class TradeRecord:
    """Record generated for every simulated trade."""

    signal: TradeSignal
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    r_multiple: Optional[float]
    status: OrderStatus
    pnl_pct: float = 0.0
    risk_amount: float = 0.0
    entry_equity: float = 0.0
    exit_equity: float = 0.0

    def as_dict(self) -> dict:
        """Return the trade as a serialisable dictionary."""

        return {
            "symbol": self.signal.symbol,
            "side": self.signal.side,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "r_multiple": self.r_multiple,
            "status": self.status,
            "pnl_pct": self.pnl_pct,
            "risk_amount": self.risk_amount,
            "entry_equity": self.entry_equity,
            "exit_equity": self.exit_equity,
        }


@dataclass(slots=True)
class BacktestResult:
    """Summary statistics for a backtest run."""

    trades: List[TradeRecord]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    win_rate: float
    average_r_multiple: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    expectancy: float
    cumulative_return: float
    cumulative_return_pct: float
    final_equity: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    equity_per_trade: pd.Series
    drawdown_abs_curve: pd.Series
    r_distribution: pd.Series
    max_consecutive_losses: int
    trade_log_path: Optional[Path] = None
    equity_curve_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        """Return the list of trades as a :class:`pandas.DataFrame`."""

        return pd.DataFrame([trade.as_dict() for trade in self.trades])

    def summary(self) -> dict[str, float]:
        """Return a dictionary with the key performance metrics of the run."""

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "average_r_multiple": self.average_r_multiple,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "expectancy": self.expectancy,
            "cumulative_return": self.cumulative_return,
            "cumulative_return_pct": self.cumulative_return_pct,
            "final_equity": self.final_equity,
        }

    def log_trades(
        self,
        logger: "TradeLogger",
        *,
        default_strategy: str = "backtest",
    ) -> None:
        """Persist the trades using the provided :class:`TradeLogger`."""

        for trade in self.trades:
            strategy_name = trade.signal.client_tag or default_strategy
            notes = (
                f"entry={trade.entry_price:.2f}, exit={trade.exit_price:.2f}, "
                f"pnl={trade.pnl:.2f}"
            )
            logger.log_trade(
                strategy=strategy_name,
                r_multiple=float(trade.r_multiple or 0.0),
                stop_loss=float(trade.signal.stop_loss or 0.0),
                take_profit=float(trade.signal.take_profit or 0.0),
                notes=notes,
                timestamp=trade.exit_time,
            )


class Backtester:
    """Simulates trade signals using historical OHLCV data."""

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        price_column: str = "close",
    ) -> None:
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise TypeError("The OHLCV data must be indexed by pandas.DatetimeIndex")
        required = {price_column, "high", "low"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"OHLCV data missing required columns: {sorted(missing)}")
        self.ohlcv = ohlcv.sort_index()
        self.price_column = price_column
        self._atr_cache: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        signals: Iterable[TradeSignal],
        *,
        settings: Optional[BacktestSettings] = None,
        initial_equity: Optional[float] = None,
        risk_per_trade_pct: Optional[float] = None,
        export_directory: Optional[Path] = None,
    ) -> BacktestResult:
        """Simulate the provided trade ``signals`` over the stored OHLCV data."""

        resolved_settings = self._resolve_settings(settings, risk=risk_per_trade_pct)
        starting_equity = float(initial_equity) if initial_equity is not None else DEFAULT_INITIAL_EQUITY

        trades: List[TradeRecord] = []
        equity_values: List[float] = []
        equity_index: List[datetime] = []
        equity_by_trade: List[Tuple[datetime, float]] = []

        equity = starting_equity
        worst_consecutive_losses = 0
        current_losses = 0

        if not self.ohlcv.empty:
            equity_index.append(self.ohlcv.index[0].to_pydatetime())
        equity_values.append(equity)

        atr_series = self._compute_atr(DEFAULT_ATR_PERIOD)
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)

        for signal in sorted_signals:
            record = self._simulate_signal(
                signal,
                equity,
                resolved_settings,
                atr_series=atr_series,
            )
            if record is None:
                continue

            trades.append(record)
            equity = record.exit_equity
            equity_values.append(equity)
            equity_index.append(record.exit_time)
            equity_by_trade.append((record.exit_time, equity))

            if record.pnl < 0:
                current_losses += 1
            else:
                current_losses = 0
            worst_consecutive_losses = max(worst_consecutive_losses, current_losses)

        equity_curve = self._build_equity_curve(equity_values, equity_index)

        (
            drawdown_curve,
            drawdown_pct_curve,
            max_drawdown,
            max_drawdown_pct,
        ) = self._compute_drawdown_metrics(equity_curve)

        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        losing_trades = sum(1 for trade in trades if trade.pnl < 0)
        win_rate = self._compute_win_rate(trades)
        average_r = self._compute_average_r(trades)
        profit_factor = self._compute_profit_factor(trades)
        expectancy = self._compute_expectancy(trades)
        final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else equity
        cumulative_return = final_equity - starting_equity
        cumulative_return_pct = (
            (final_equity / starting_equity - 1) * 100 if starting_equity else 0.0
        )

        r_distribution = self._build_r_distribution(trades)
        equity_per_trade_series = (
            pd.Series(
                [value for _, value in equity_by_trade],
                index=pd.to_datetime([ts for ts, _ in equity_by_trade]),
                name="equity_by_trade",
            )
            if equity_by_trade
            else pd.Series(dtype=float, name="equity_by_trade")
        )

        trade_log_path: Optional[Path] = None
        equity_curve_path: Optional[Path] = None
        metrics_path: Optional[Path] = None
        if export_directory:
            trade_log_path, equity_curve_path, metrics_path = self._export_results(
                export_directory,
                trades,
                equity_curve,
                r_distribution,
                starting_equity,
                average_r,
                profit_factor,
            )

        metadata = {
            "risk_per_trade_pct": resolved_settings.risk_per_trade_pct,
            "rr_ratio": resolved_settings.rr_ratio,
            "sl_ratio": resolved_settings.sl_ratio,
        }
        if metrics_path is not None:
            metadata["metrics_path"] = str(metrics_path)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_pct_curve,
            win_rate=win_rate,
            average_r_multiple=average_r,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            expectancy=expectancy,
            cumulative_return=cumulative_return,
            cumulative_return_pct=cumulative_return_pct,
            final_equity=final_equity,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            equity_per_trade=equity_per_trade_series,
            drawdown_abs_curve=drawdown_curve,
            r_distribution=r_distribution,
            max_consecutive_losses=worst_consecutive_losses,
            trade_log_path=trade_log_path,
            equity_curve_path=equity_curve_path,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_settings(
        self,
        settings: Optional[BacktestSettings],
        *,
        risk: Optional[float],
    ) -> BacktestSettings:
        base = settings or BacktestSettings()
        if risk is not None:
            base = replace(base, risk_per_trade_pct=risk)
        return base

    def _compute_atr(self, period: int) -> pd.Series:
        if self._atr_cache is not None:
            return self._atr_cache

        high = self.ohlcv["high"].astype(float)
        low = self.ohlcv["low"].astype(float)
        close = self.ohlcv[self.price_column].astype(float)

        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        self._atr_cache = atr
        return atr

    @staticmethod
    def _lookup_series_value(series: pd.Series, timestamp: datetime) -> float:
        if series.empty:
            return 0.0
        ts = pd.Timestamp(timestamp)
        if ts in series.index:
            return float(series.loc[ts])
        before = series.loc[:ts]
        if before.empty:
            return float(series.iloc[0])
        return float(before.iloc[-1])

    def _prepare_signal_levels(
        self,
        signal: TradeSignal,
        entry_price: float,
        atr_value: float,
        *,
        settings: BacktestSettings,
    ) -> Tuple[TradeSignal, float]:
        base_distance = 0.0

        if signal.stop_loss is not None:
            base_distance = abs(entry_price - float(signal.stop_loss))
        elif signal.take_profit is not None and settings.rr_ratio > 0:
            base_distance = abs(float(signal.take_profit) - entry_price) / settings.rr_ratio
        elif atr_value > 0:
            base_distance = atr_value

        if base_distance <= 0:
            base_distance = entry_price * 0.01

        stop_distance = settings.sl_ratio * base_distance
        take_distance = settings.rr_ratio * base_distance

        if signal.side == "BUY":
            resolved_stop = entry_price - stop_distance
            resolved_take = entry_price + take_distance
        else:
            resolved_stop = entry_price + stop_distance
            resolved_take = entry_price - take_distance

        adjusted_signal = replace(
            signal,
            stop_loss=resolved_stop,
            take_profit=resolved_take,
        )
        return adjusted_signal, abs(entry_price - resolved_stop)

    def _compute_commissions(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        settings: BacktestSettings,
    ) -> float:
        rate = settings.commission_pct / 100
        if rate <= 0:
            return 0.0
        trade_value = entry_price + exit_price
        return trade_value * quantity * rate

    def _build_equity_curve(
        self, equity_values: List[float], equity_index: List[datetime]
    ) -> pd.Series:
        if equity_index and len(equity_index) == len(equity_values):
            index = pd.to_datetime(equity_index)
            return pd.Series(equity_values, index=index, name="equity")
        return pd.Series(equity_values, name="equity")

    def _compute_drawdown_metrics(
        self, equity_curve: pd.Series
    ) -> tuple[pd.Series, pd.Series, float, float]:
        if equity_curve.empty:
            empty = pd.Series(dtype=float)
            return empty, empty, 0.0, 0.0
        equity_cummax = equity_curve.cummax()
        drawdown_abs = (equity_cummax - equity_curve).fillna(0.0)
        drawdown_pct = (
            (drawdown_abs / equity_cummax.replace(0.0, np.nan)).fillna(0.0)
        )
        max_drawdown = float(drawdown_abs.max()) if not drawdown_abs.empty else 0.0
        max_drawdown_pct = (
            float(drawdown_pct.max()) * 100 if not drawdown_pct.empty else 0.0
        )
        return drawdown_abs, drawdown_pct, max_drawdown, max_drawdown_pct

    def _build_r_distribution(self, trades: Sequence[TradeRecord]) -> pd.Series:
        r_values = [
            trade.r_multiple for trade in trades if trade.r_multiple is not None
        ]
        if not r_values:
            return pd.Series(dtype=float, name="r_multiple")
        return pd.Series(r_values, name="r_multiple")

    def _export_results(
        self,
        directory: Path,
        trades: Sequence[TradeRecord],
        equity_curve: pd.Series,
        r_distribution: pd.Series,
        starting_equity: float,
        average_r: float,
        profit_factor: float,
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        trade_path = directory / f"trades-{timestamp}.csv"
        equity_path = directory / f"equity-{timestamp}.csv"
        summary_path = directory / f"summary-{timestamp}.json"
        r_path = directory / f"r-multiples-{timestamp}.csv"
        metrics_path = directory / f"asd-{timestamp}.csv"
        latest_metrics_path = directory / "asd.csv"

        trades_frame = pd.DataFrame([trade.as_dict() for trade in trades])
        trades_frame.to_csv(trade_path, index=False)
        equity_curve.to_csv(equity_path, header=True)
        r_distribution.to_csv(r_path, header=True)

        final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else starting_equity
        summary = {
            "initial_equity": starting_equity,
            "final_equity": final_equity,
            "total_trades": len(trades),
            "generated_at": timestamp,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        r_values = [trade.r_multiple for trade in trades if trade.r_multiple is not None]
        distribution_metrics = trade_distribution_metrics(r_values)
        metrics_row = {
            "Initial Equity": starting_equity,
            "Final Equity": final_equity,
            "Total Trades": len(trades),
            "Winrate %": distribution_metrics["Winrate"] * 100,
            "Average R Multiple": average_r,
            "Average Win (R)": distribution_metrics["Average Win (R)"],
            "Average Loss (R)": distribution_metrics["Average Loss (R)"],
            "RR Effective": distribution_metrics["RR Effective"],
            "Breakeven Winrate %": distribution_metrics["Breakeven Winrate %"],
            "Expectancy (R)": distribution_metrics["Expectancy (R)"],
            "Profit Factor": profit_factor,
        }
        metrics_df = pd.DataFrame([metrics_row])
        metrics_df.to_csv(metrics_path, index=False)
        metrics_df.to_csv(latest_metrics_path, index=False)
        return trade_path, equity_path, metrics_path

    def _simulate_signal(
        self,
        signal: TradeSignal,
        account_equity: float,
        settings: BacktestSettings,
        *,
        atr_series: pd.Series,
    ) -> Optional[TradeRecord]:
        entry_position = int(self.ohlcv.index.searchsorted(signal.timestamp))
        if entry_position >= len(self.ohlcv.index):
            return None

        entry_df = self.ohlcv.iloc[entry_position:]
        if entry_df.empty:
            return None

        if signal.entry_type == "market":
            entry_time = entry_df.index[0]
            entry_price = float(entry_df.iloc[0][self.price_column])
        else:
            entry_info = self._find_limit_entry(signal, entry_position)
            if entry_info is None:
                return None
            entry_time, entry_price, entry_position = entry_info
            entry_df = self.ohlcv.iloc[entry_position:]

        atr_value = float(self._lookup_series_value(atr_series, entry_time))

        adjusted_signal, stop_distance = self._prepare_signal_levels(
            signal,
            entry_price,
            atr_value,
            settings=settings,
        )

        exit_time, exit_price, status = self._simulate_exit(
            adjusted_signal,
            entry_price,
            entry_df,
        )

        target_risk = max(account_equity * settings.risk_per_trade_pct / 100, 0.0)
        quantity, risk_amount = self._position_size(
            entry_price,
            adjusted_signal.stop_loss,
            adjusted_signal.quantity,
            target_risk,
            stop_distance=stop_distance,
        )
        if quantity <= 0:
            return None

        pnl = self._compute_pnl(adjusted_signal.side, entry_price, exit_price, quantity)

        fees = self._compute_commissions(
            entry_price,
            exit_price,
            quantity,
            settings,
        )
        pnl -= fees

        r_multiple = self._compute_r_multiple(
            adjusted_signal.side,
            entry_price,
            exit_price,
            adjusted_signal.stop_loss,
        )
        if (r_multiple is None or np.isnan(r_multiple)) and risk_amount > 0:
            r_multiple = pnl / risk_amount

        entry_equity = account_equity
        exit_equity = account_equity + pnl
        pnl_pct = pnl / account_equity if account_equity > 0 else 0.0

        return TradeRecord(
            signal=adjusted_signal,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            r_multiple=r_multiple,
            status=status,
            pnl_pct=pnl_pct,
            risk_amount=risk_amount,
            entry_equity=entry_equity,
            exit_equity=exit_equity,
        )

    @staticmethod
    def _position_size(
        entry_price: float,
        stop_price: Optional[float],
        requested_quantity: float,
        target_risk: float,
        *,
        stop_distance: float,
    ) -> tuple[float, float]:
        quantity = max(float(requested_quantity), 0.0)
        actual_risk = max(float(target_risk), 0.0)

        if target_risk <= 0:
            return quantity if quantity > 0 else 0.0, 0.0

        if stop_price is not None:
            risk_per_unit = abs(entry_price - float(stop_price))
            if risk_per_unit > 0:
                desired_quantity = target_risk / risk_per_unit
                if quantity > 0:
                    quantity = min(quantity, desired_quantity)
                    actual_risk = risk_per_unit * quantity
                else:
                    quantity = desired_quantity
                    actual_risk = target_risk
                return quantity, actual_risk

        if stop_distance > 0:
            risk_per_unit = stop_distance
            desired_quantity = target_risk / risk_per_unit
            if quantity > 0:
                quantity = min(quantity, desired_quantity)
                actual_risk = risk_per_unit * quantity
            else:
                quantity = desired_quantity
                actual_risk = target_risk
            return quantity, actual_risk

        if quantity > 0:
            return quantity, actual_risk if stop_price is not None else target_risk

        if entry_price > 0:
            qty = target_risk / entry_price
            return qty, target_risk

        return 0.0, 0.0

    def _find_limit_entry(
        self, signal: TradeSignal, start_position: int
    ) -> Optional[tuple[datetime, float, int]]:
        if signal.entry_price is None:
            raise ValueError("Limit signals require an entry_price")

        for idx in range(start_position, len(self.ohlcv.index)):
            row = self.ohlcv.iloc[idx]
            high = float(row["high"])
            low = float(row["low"])
            current_index = self.ohlcv.index[idx]
            if signal.side == "BUY" and low <= signal.entry_price:
                return current_index, float(signal.entry_price), idx
            if signal.side == "SELL" and high >= signal.entry_price:
                return current_index, float(signal.entry_price), idx
        return None

    def _simulate_exit(
        self,
        signal: TradeSignal,
        entry_price: float,
        entry_df: pd.DataFrame,
    ) -> tuple[datetime, float, OrderStatus]:
        stop_price = float(signal.stop_loss) if signal.stop_loss is not None else None
        take_profit = (
            float(signal.take_profit) if signal.take_profit is not None else None
        )
        for _, row in entry_df.iterrows():
            current_time = row.name
            high = float(row["high"])
            low = float(row["low"])

            if signal.side == "BUY":
                if stop_price is not None and low <= stop_price:
                    return current_time, float(stop_price), OrderStatus.FILLED
                if take_profit is not None and high >= take_profit:
                    return current_time, float(take_profit), OrderStatus.FILLED
            else:
                if stop_price is not None and high >= stop_price:
                    return current_time, float(stop_price), OrderStatus.FILLED
                if take_profit is not None and low <= take_profit:
                    return current_time, float(take_profit), OrderStatus.FILLED

        last_row = entry_df.iloc[-1]
        final_price = float(last_row[self.price_column])
        return (entry_df.index[-1], final_price, OrderStatus.EXPIRED)

    @staticmethod
    def _compute_pnl(
        side: str, entry_price: float, exit_price: float, quantity: float
    ) -> float:
        direction = 1 if side == "BUY" else -1
        return direction * (exit_price - entry_price) * quantity

    @staticmethod
    def _compute_r_multiple(
        side: str, entry_price: float, exit_price: float, stop_price: Optional[float]
    ) -> Optional[float]:
        if stop_price is None:
            return None
        risk = entry_price - stop_price if side == "BUY" else stop_price - entry_price
        if risk == 0:
            return None
        reward = exit_price - entry_price if side == "BUY" else entry_price - exit_price
        return reward / risk

    @staticmethod
    def _compute_win_rate(trades: Sequence[TradeRecord]) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for trade in trades if trade.pnl > 0)
        return wins / len(trades)

    @staticmethod
    def _compute_average_r(trades: Sequence[TradeRecord]) -> float:
        r_values = [
            trade.r_multiple for trade in trades if trade.r_multiple is not None
        ]
        if not r_values:
            return 0.0
        return float(np.mean(r_values))

    @staticmethod
    def _compute_profit_factor(trades: Sequence[TradeRecord]) -> float:
        if not trades:
            return 0.0
        gains = [trade.pnl for trade in trades if trade.pnl > 0]
        losses = [abs(trade.pnl) for trade in trades if trade.pnl < 0]
        gross_profit = float(np.sum(gains)) if gains else 0.0
        gross_loss = float(np.sum(losses)) if losses else 0.0
        if gross_loss == 0.0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def _compute_expectancy(trades: Sequence[TradeRecord]) -> float:
        r_values = [
            trade.r_multiple for trade in trades if trade.r_multiple is not None
        ]
        if not r_values:
            return 0.0
        wins = [r for r in r_values if r > 0]
        losses = [r for r in r_values if r < 0]
        win_rate = len(wins) / len(r_values)
        loss_rate = len(losses) / len(r_values)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        return win_rate * avg_win + loss_rate * avg_loss


def aggregate_backtest_results(
    results: Dict[str, BacktestResult]
) -> BacktestResult:
    """Combine several BacktestResult instances into a single report."""

    if not results:
        raise ValueError("results must contain at least one BacktestResult")

    component_curves = []
    aggregated_trades: List[TradeRecord] = []
    for name, result in results.items():
        curve = result.equity_curve.rename(name)
        component_curves.append(curve)
        for trade in result.trades:
            signal = trade.signal
            if signal.client_tag != name:
                signal = replace(signal, client_tag=name)
            aggregated_trades.append(
                replace(
                    trade,
                    signal=signal,
                )
            )

    combined_equity = pd.concat(component_curves, axis=1).sort_index()
    combined_equity.index = pd.to_datetime(combined_equity.index)
    combined_equity = combined_equity.ffill().bfill()
    total_equity = combined_equity.sum(axis=1)

    if total_equity.empty:
        total_equity = pd.Series([0.0], name="equity")

    helper = Backtester(
        pd.DataFrame(
            {
                "close": total_equity,
                "high": total_equity,
                "low": total_equity,
            }
        )
    )
    drawdown_abs, drawdown_pct, max_dd, max_dd_pct = helper._compute_drawdown_metrics(
        total_equity
    )

    aggregated_trades.sort(key=lambda trade: trade.exit_time)
    starting_equity = float(total_equity.iloc[0])
    current_equity = starting_equity
    equity_history: List[Tuple[datetime, float]] = []
    worst_streak = 0
    losing_streak = 0
    normalised_trades: List[TradeRecord] = []
    for trade in aggregated_trades:
        entry_equity = current_equity
        exit_equity = current_equity + trade.pnl
        current_equity = exit_equity
        equity_history.append((trade.exit_time, current_equity))
        if trade.pnl < 0:
            losing_streak += 1
        else:
            losing_streak = 0
        worst_streak = max(worst_streak, losing_streak)
        normalised_trades.append(
            replace(
                trade,
                entry_equity=entry_equity,
                exit_equity=exit_equity,
            )
        )

    win_rate = Backtester._compute_win_rate(normalised_trades)
    average_r = Backtester._compute_average_r(normalised_trades)
    profit_factor = Backtester._compute_profit_factor(normalised_trades)
    expectancy = Backtester._compute_expectancy(normalised_trades)

    final_equity = float(total_equity.iloc[-1])
    cumulative_return = final_equity - starting_equity
    cumulative_return_pct = (
        (final_equity / starting_equity - 1) * 100 if starting_equity else 0.0
    )

    equity_curve = total_equity.rename("equity")
    equity_per_trade = (
        pd.Series(
            [value for _, value in equity_history],
            index=pd.to_datetime([ts for ts, _ in equity_history]),
            name="equity_by_trade",
        )
        if equity_history
        else pd.Series(dtype=float, name="equity_by_trade")
    )
    r_distribution = helper._build_r_distribution(normalised_trades)

    return BacktestResult(
        trades=normalised_trades,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_pct,
        win_rate=win_rate,
        average_r_multiple=average_r,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        expectancy=expectancy,
        cumulative_return=cumulative_return,
        cumulative_return_pct=cumulative_return_pct,
        final_equity=final_equity,
        total_trades=len(normalised_trades),
        winning_trades=sum(1 for trade in normalised_trades if trade.pnl > 0),
        losing_trades=sum(1 for trade in normalised_trades if trade.pnl < 0),
        equity_per_trade=equity_per_trade,
        drawdown_abs_curve=drawdown_abs,
        r_distribution=r_distribution,
        max_consecutive_losses=worst_streak,
        trade_log_path=None,
        equity_curve_path=None,
        metadata={"strategy_count": len(results)},
    )
