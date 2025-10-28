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
from datetime import datetime, time
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .types import OrderStatus, TradeSignal

if TYPE_CHECKING:
    from analytics.trade_logger import TradeLogger


def _validate_percentage(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be greater or equal than zero")
    return float(value)


@dataclass(slots=True)
class BacktestSettings:
    """Container for the configurable aspects of a backtest run."""

    initial_equity: float = 1_000.0
    risk_per_trade_pct: float = 1.0
    commission_pct: float = 0.0
    maker_commission_pct: Optional[float] = None
    taker_commission_pct: Optional[float] = None
    slippage_pct: float = 0.0
    use_trailing_stop: bool = False
    trailing_activation_r: float = 1.0
    trailing_method: str = "atr"
    trailing_atr_multiplier: float = 1.0
    trailing_percent: float = 0.5
    default_stop_atr_multiplier: float = 1.0
    atr_period: int = 14
    max_daily_loss_pct: float = 10.0
    max_consecutive_losses: int = 10
    use_session_limits: bool = False
    session_start: Optional[time] = None
    session_end: Optional[time] = None
    volatility_atr_threshold: Optional[float] = None
    min_volume_ratio: float = 0.5
    export_directory: Optional[Path] = None
    strategies: Optional[Dict[str, bool]] = None

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        _validate_percentage("risk_per_trade_pct", self.risk_per_trade_pct)
        if self.maker_commission_pct is not None:
            _validate_percentage("maker_commission_pct", self.maker_commission_pct)
        if self.taker_commission_pct is not None:
            _validate_percentage("taker_commission_pct", self.taker_commission_pct)
        _validate_percentage("commission_pct", self.commission_pct)
        _validate_percentage("slippage_pct", self.slippage_pct)
        _validate_percentage("max_daily_loss_pct", self.max_daily_loss_pct)
        if self.max_consecutive_losses <= 0:
            raise ValueError("max_consecutive_losses must be positive")
        if self.trailing_method not in {"atr", "percent"}:
            raise ValueError("trailing_method must be 'atr' or 'percent'")
        if self.trailing_activation_r < 0:
            raise ValueError("trailing_activation_r must be non-negative")
        if self.trailing_method == "percent" and self.trailing_percent <= 0:
            raise ValueError("trailing_percent must be positive when using percentage trailing")
        if self.trailing_method == "atr" and self.trailing_atr_multiplier <= 0:
            raise ValueError("trailing_atr_multiplier must be positive when using ATR trailing")
        if self.default_stop_atr_multiplier <= 0:
            raise ValueError("default_stop_atr_multiplier must be positive")
        if self.atr_period <= 1:
            raise ValueError("atr_period must be greater than 1")
        if self.min_volume_ratio < 0:
            raise ValueError("min_volume_ratio must be non-negative")


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
    metadata: Dict[str, float] = field(default_factory=dict)

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
    ) -> BacktestResult:
        """Simulate the provided trade ``signals`` over the stored OHLCV data."""

        resolved_settings = self._resolve_settings(
            settings, initial_equity=initial_equity, risk=risk_per_trade_pct
        )

        trades: List[TradeRecord] = []
        equity_values: List[float] = []
        equity_index: List[datetime] = []
        equity_by_trade: List[Tuple[datetime, float]] = []

        equity = float(resolved_settings.initial_equity)
        consecutive_losses = 0
        worst_consecutive_losses = 0
        daily_stats: Dict[pd.Timestamp, Dict[str, float]] = {}

        if not self.ohlcv.empty:
            equity_index.append(self.ohlcv.index[0].to_pydatetime())
        equity_values.append(equity)

        atr_series = self._compute_atr(resolved_settings.atr_period)

        sorted_signals = sorted(signals, key=lambda s: s.timestamp)

        for signal in sorted_signals:
            if not self._is_signal_enabled(signal, resolved_settings):
                continue

            if resolved_settings.use_session_limits and not self._within_session(
                signal.timestamp, resolved_settings
            ):
                continue

            signal_day = pd.Timestamp(signal.timestamp).normalize()
            stats = daily_stats.setdefault(
                signal_day, {"start_equity": equity, "pnl": 0.0, "stop": False}
            )
            if stats["stop"]:
                continue
            if consecutive_losses >= resolved_settings.max_consecutive_losses:
                break

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

            pnl = record.pnl
            if pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
            worst_consecutive_losses = max(worst_consecutive_losses, consecutive_losses)

            stats["pnl"] += pnl
            if stats["pnl"] <= -resolved_settings.max_daily_loss_pct / 100 * stats[
                "start_equity"
            ]:
                stats["stop"] = True

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
        cumulative_return = final_equity - float(resolved_settings.initial_equity)
        cumulative_return_pct = (
            (final_equity / float(resolved_settings.initial_equity) - 1) * 100
            if resolved_settings.initial_equity
            else 0.0
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
        if resolved_settings.export_directory:
            trade_log_path, equity_curve_path = self._export_results(
                resolved_settings.export_directory,
                trades,
                equity_curve,
                r_distribution,
                resolved_settings,
            )

        metadata = {
            "max_daily_loss_pct": resolved_settings.max_daily_loss_pct,
            "risk_per_trade_pct": resolved_settings.risk_per_trade_pct,
        }

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
        initial_equity: Optional[float],
        risk: Optional[float],
    ) -> BacktestSettings:
        if settings is None:
            base = BacktestSettings()
        else:
            base = settings

        if initial_equity is not None or risk is not None:
            base = replace(
                base,
                **{
                    k: v
                    for k, v in {
                        "initial_equity": initial_equity
                        if initial_equity is not None
                        else base.initial_equity,
                        "risk_per_trade_pct": risk
                        if risk is not None
                        else base.risk_per_trade_pct,
                    }.items()
                    if v is not None
                },
            )
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

    def _is_signal_enabled(
        self, signal: TradeSignal, settings: BacktestSettings
    ) -> bool:
        if not settings.strategies:
            return True
        if signal.client_tag is None:
            return True
        enabled = settings.strategies.get(signal.client_tag)
        return bool(enabled) if enabled is not None else True

    def _within_session(
        self, timestamp: datetime, settings: BacktestSettings
    ) -> bool:
        if not settings.use_session_limits:
            return True
        ts_time = timestamp.time()
        if settings.session_start and ts_time < settings.session_start:
            return False
        if settings.session_end and ts_time > settings.session_end:
            return False
        return True

    def _prepare_signal_levels(
        self,
        signal: TradeSignal,
        entry_price: float,
        atr_value: float,
        *,
        settings: BacktestSettings,
    ) -> Tuple[TradeSignal, float]:
        stop_price = signal.stop_loss
        stop_distance = 0.0
        if stop_price is not None:
            stop_distance = abs(entry_price - float(stop_price))

        if stop_distance <= 0:
            stop_distance = atr_value * settings.default_stop_atr_multiplier

        if stop_distance <= 0:
            stop_distance = entry_price * 0.01  # Fallback 1%

        if signal.side == "BUY":
            resolved_stop = entry_price - stop_distance
            resolved_take = entry_price + 2 * stop_distance
        else:
            resolved_stop = entry_price + stop_distance
            resolved_take = entry_price - 2 * stop_distance

        if signal.take_profit is not None:
            resolved_take = float(signal.take_profit)

        trailing_delta: Optional[float] = signal.trailing_delta
        if settings.use_trailing_stop:
            if settings.trailing_method == "atr":
                trailing_delta = atr_value * settings.trailing_atr_multiplier
            else:
                trailing_delta = entry_price * settings.trailing_percent / 100

        adjusted_signal = replace(
            signal,
            stop_loss=resolved_stop,
            take_profit=resolved_take,
            trailing_delta=trailing_delta,
        )
        return adjusted_signal, stop_distance

    @staticmethod
    def _apply_slippage(
        price: float, side: str, settings: BacktestSettings
    ) -> float:
        slippage = settings.slippage_pct / 100
        if slippage <= 0:
            return price
        adjustment = price * slippage
        return price + adjustment if side == "BUY" else price - adjustment

    def _compute_commissions(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        settings: BacktestSettings,
    ) -> float:
        rate_entry = settings.maker_commission_pct
        if rate_entry is None:
            rate_entry = settings.commission_pct
        rate_exit = settings.taker_commission_pct
        if rate_exit is None:
            rate_exit = settings.commission_pct
        entry_fee = entry_price * quantity * rate_entry / 100 if rate_entry else 0.0
        exit_fee = exit_price * quantity * rate_exit / 100 if rate_exit else 0.0
        return entry_fee + exit_fee

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
        settings: BacktestSettings,
    ) -> tuple[Optional[Path], Optional[Path]]:
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        trade_path = directory / f"trades-{timestamp}.csv"
        equity_path = directory / f"equity-{timestamp}.csv"
        summary_path = directory / f"summary-{timestamp}.json"
        r_path = directory / f"r-multiples-{timestamp}.csv"

        trades_frame = pd.DataFrame([trade.as_dict() for trade in trades])
        trades_frame.to_csv(trade_path, index=False)
        equity_curve.to_csv(equity_path, header=True)
        r_distribution.to_csv(r_path, header=True)

        summary = {
            "initial_equity": settings.initial_equity,
            "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else settings.initial_equity,
            "total_trades": len(trades),
            "generated_at": timestamp,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return trade_path, equity_path

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
        if settings.volatility_atr_threshold is not None and atr_value < settings.volatility_atr_threshold:
            return None

        if "volume" in entry_df.columns:
            lookback = max(1, settings.atr_period)
            volume_window = self.ohlcv.iloc[max(entry_position - lookback, 0):entry_position][
                "volume"
            ]
            avg_volume = float(volume_window.mean()) if not volume_window.empty else float(
                entry_df.iloc[0]["volume"]
            )
            current_volume = float(entry_df.iloc[0]["volume"])
            if avg_volume > 0 and current_volume < settings.min_volume_ratio * avg_volume:
                return None

        adjusted_signal, stop_distance = self._prepare_signal_levels(
            signal,
            entry_price,
            atr_value,
            settings=settings,
        )

        entry_price = self._apply_slippage(entry_price, adjusted_signal.side, settings)

        exit_time, exit_price, status = self._simulate_exit(
            adjusted_signal,
            entry_price,
            entry_df,
            settings=settings,
            stop_distance=stop_distance,
            atr_series=atr_series,
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
        *,
        settings: BacktestSettings,
        stop_distance: float,
        atr_series: pd.Series,
    ) -> tuple[datetime, float, OrderStatus]:
        stop_price = float(signal.stop_loss) if signal.stop_loss is not None else None
        take_profit = (
            float(signal.take_profit) if signal.take_profit is not None else None
        )
        trailing_active = False
        trailing_stop: Optional[float] = stop_price
        activation_price = None
        if stop_distance > 0 and settings.trailing_activation_r > 0:
            if signal.side == "BUY":
                activation_price = entry_price + stop_distance * settings.trailing_activation_r
            else:
                activation_price = entry_price - stop_distance * settings.trailing_activation_r

        for _, row in entry_df.iterrows():
            current_time = row.name
            high = float(row["high"])
            low = float(row["low"])

            if settings.use_trailing_stop and activation_price is not None and not trailing_active:
                if signal.side == "BUY" and high >= activation_price:
                    trailing_active = True
                elif signal.side == "SELL" and low <= activation_price:
                    trailing_active = True

            if settings.use_trailing_stop and trailing_active:
                if settings.trailing_method == "atr":
                    atr_value = self._lookup_series_value(atr_series, current_time)
                    trailing_delta = atr_value * settings.trailing_atr_multiplier
                else:
                    trailing_delta = entry_price * settings.trailing_percent / 100

                if signal.side == "BUY":
                    candidate = high - trailing_delta
                    trailing_stop = max(trailing_stop or -np.inf, candidate)
                else:
                    candidate = low + trailing_delta
                    trailing_stop = min(trailing_stop or np.inf, candidate)

            if signal.side == "BUY":
                stop_candidate = trailing_stop if trailing_stop is not None else stop_price
                if stop_candidate is not None and low <= stop_candidate:
                    exit_price = self._apply_slippage(
                        float(stop_candidate), "SELL", settings
                    )
                    return current_time, exit_price, OrderStatus.FILLED
                if take_profit is not None and high >= take_profit:
                    exit_price = self._apply_slippage(
                        float(take_profit), "SELL", settings
                    )
                    return current_time, exit_price, OrderStatus.FILLED
            else:
                stop_candidate = trailing_stop if trailing_stop is not None else stop_price
                if stop_candidate is not None and high >= stop_candidate:
                    exit_price = self._apply_slippage(
                        float(stop_candidate), "BUY", settings
                    )
                    return current_time, exit_price, OrderStatus.FILLED
                if take_profit is not None and low <= take_profit:
                    exit_price = self._apply_slippage(
                        float(take_profit), "BUY", settings
                    )
                    return current_time, exit_price, OrderStatus.FILLED

        last_row = entry_df.iloc[-1]
        final_price = self._apply_slippage(
            float(last_row[self.price_column]),
            "SELL" if signal.side == "BUY" else "BUY",
            settings,
        )
        return (
            entry_df.index[-1],
            final_price,
            OrderStatus.EXPIRED,
        )

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
