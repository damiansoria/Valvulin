"""Historical backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .types import OrderStatus, TradeSignal


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


class Backtester:
    """Simulates trade signals using historical OHLCV data."""

    def __init__(self, ohlcv: pd.DataFrame, price_column: str = "close") -> None:
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise TypeError("The OHLCV data must be indexed by pandas.DatetimeIndex")
        required = {price_column, "high", "low"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"OHLCV data missing required columns: {sorted(missing)}")
        self.ohlcv = ohlcv.sort_index()
        self.price_column = price_column

    def run(
        self,
        signals: Iterable[TradeSignal],
        *,
        initial_equity: float = 1_000.0,
        risk_per_trade_pct: float = 1.0,
    ) -> BacktestResult:
        """Simulate the provided trade ``signals`` over the stored OHLCV data."""

        trades: List[TradeRecord] = []
        equity_values: List[float] = []
        equity_index: List[datetime] = []

        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if risk_per_trade_pct < 0:
            raise ValueError("risk_per_trade_pct must be non-negative")

        equity = float(initial_equity)
        if not self.ohlcv.empty:
            equity_index.append(self.ohlcv.index[0].to_pydatetime())
            equity_values.append(equity)
        else:
            equity_values.append(equity)

        risk_fraction = float(risk_per_trade_pct) / 100

        for signal in signals:
            record = self._simulate_signal(signal, equity, risk_fraction)
            if record is None:
                continue
            trades.append(record)
            equity = record.exit_equity
            equity_values.append(equity)
            equity_index.append(record.exit_time)

        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        losing_trades = sum(1 for trade in trades if trade.pnl < 0)
        win_rate = self._compute_win_rate(trades)
        average_r = self._compute_average_r(trades)
        profit_factor = self._compute_profit_factor(trades)
        expectancy = self._compute_expectancy(trades)

        if equity_index and len(equity_index) == len(equity_values):
            equity_curve = pd.Series(
                equity_values, index=pd.to_datetime(equity_index), name="equity"
            )
        else:
            equity_curve = pd.Series(equity_values, name="equity")

        if equity_curve.empty:
            equity_curve = pd.Series([initial_equity], name="equity")

        equity_cummax = equity_curve.cummax()
        drawdown_curve = (equity_cummax - equity_curve).fillna(0.0)
        drawdown_pct_curve = (
            (drawdown_curve / equity_cummax.replace(0.0, np.nan)).fillna(0.0)
        )

        max_drawdown = float(drawdown_curve.max()) if not drawdown_curve.empty else 0.0
        max_drawdown_pct = (
            float(drawdown_pct_curve.max()) * 100 if not drawdown_pct_curve.empty else 0.0
        )
        final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else equity
        cumulative_return = final_equity - float(initial_equity)
        cumulative_return_pct = (
            (final_equity / float(initial_equity) - 1) * 100 if initial_equity else 0.0
        )

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
        )

    def _simulate_signal(
        self, signal: TradeSignal, account_equity: float, risk_fraction: float
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

        exit_time, exit_price, status = self._simulate_exit(
            signal, entry_price, entry_df
        )

        target_risk = max(account_equity * risk_fraction, 0.0)
        quantity, risk_amount = self._position_size(
            entry_price, signal.stop_loss, signal.quantity, target_risk
        )

        pnl = self._compute_pnl(signal.side, entry_price, exit_price, quantity)
        r_multiple = self._compute_r_multiple(
            signal.side, entry_price, exit_price, signal.stop_loss
        )
        if r_multiple is None and risk_amount > 0:
            r_multiple = pnl / risk_amount

        entry_equity = account_equity
        exit_equity = account_equity + pnl
        pnl_pct = pnl / account_equity if account_equity > 0 else 0.0

        return TradeRecord(
            signal=signal,
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
        self, signal: TradeSignal, entry_price: float, entry_df: pd.DataFrame
    ) -> tuple[datetime, float, OrderStatus]:
        stop_price = signal.stop_loss
        take_profit = signal.take_profit
        trailing_delta = signal.trailing_delta
        trailing_stop = None
        if trailing_delta is not None:
            trailing_stop = (
                entry_price - trailing_delta
                if signal.side == "BUY"
                else entry_price + trailing_delta
            )

        for _, row in entry_df.iterrows():
            current_time = row.name
            high = float(row["high"])
            low = float(row["low"])

            if trailing_delta is not None:
                if signal.side == "BUY":
                    trailing_stop = max(trailing_stop, high - trailing_delta)
                else:
                    trailing_stop = min(trailing_stop, low + trailing_delta)

            if signal.side == "BUY":
                if stop_price is not None and low <= stop_price:
                    return current_time, float(stop_price), OrderStatus.FILLED
                if take_profit is not None and high >= take_profit:
                    return current_time, float(take_profit), OrderStatus.FILLED
                if trailing_stop is not None and low <= trailing_stop:
                    return current_time, float(trailing_stop), OrderStatus.FILLED
            else:
                if stop_price is not None and high >= stop_price:
                    return current_time, float(stop_price), OrderStatus.FILLED
                if take_profit is not None and low <= take_profit:
                    return current_time, float(take_profit), OrderStatus.FILLED
                if trailing_stop is not None and high >= trailing_stop:
                    return current_time, float(trailing_stop), OrderStatus.FILLED

        last_row = entry_df.iloc[-1]
        return (
            entry_df.index[-1],
            float(last_row[self.price_column]),
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
