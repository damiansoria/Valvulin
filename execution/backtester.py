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
        }


@dataclass(slots=True)
class BacktestResult:
    """Summary statistics for a backtest run."""

    trades: List[TradeRecord]
    equity_curve: pd.Series
    win_rate: float
    average_r_multiple: float
    profit_factor: float
    max_drawdown: float
    expectancy: float
    cumulative_return: float

    def trades_frame(self) -> pd.DataFrame:
        """Return the list of trades as a :class:`pandas.DataFrame`."""

        return pd.DataFrame([trade.as_dict() for trade in self.trades])

    def summary(self) -> dict[str, float]:
        """Return a dictionary with the key performance metrics of the run."""

        return {
            "win_rate": self.win_rate,
            "average_r_multiple": self.average_r_multiple,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "expectancy": self.expectancy,
            "cumulative_return": self.cumulative_return,
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

    def run(self, signals: Iterable[TradeSignal]) -> BacktestResult:
        """Simulate the provided trade ``signals`` over the stored OHLCV data."""

        trades: List[TradeRecord] = []
        equity_values: List[float] = []
        equity_index: List[datetime] = []
        equity = 0.0

        for signal in signals:
            record = self._simulate_signal(signal)
            if record is None:
                continue
            trades.append(record)
            equity += record.pnl
            equity_values.append(equity)
            equity_index.append(record.exit_time)

        win_rate = self._compute_win_rate(trades)
        average_r = self._compute_average_r(trades)
        profit_factor = self._compute_profit_factor(trades)
        expectancy = self._compute_expectancy(trades)
        max_drawdown = self._compute_drawdown(equity_values)
        cumulative_return = equity_values[-1] if equity_values else 0.0

        equity_curve = pd.Series(
            equity_values, index=pd.to_datetime(equity_index), name="equity"
        )

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            win_rate=win_rate,
            average_r_multiple=average_r,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            expectancy=expectancy,
            cumulative_return=cumulative_return,
        )

    def _simulate_signal(self, signal: TradeSignal) -> Optional[TradeRecord]:
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
        quantity = signal.quantity
        pnl = self._compute_pnl(signal.side, entry_price, exit_price, quantity)
        r_multiple = self._compute_r_multiple(
            signal.side, entry_price, exit_price, signal.stop_loss
        )

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
        )

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
    def _compute_drawdown(equity_curve: Sequence[float]) -> float:
        if not equity_curve:
            return 0.0
        if isinstance(equity_curve, np.ndarray):
            equity_array = equity_curve.astype(float, copy=False)
        else:
            equity_array = np.asarray(equity_curve, dtype=float)
        equity_array = np.concatenate(([0.0], equity_array))
        peaks = np.maximum.accumulate(equity_array)
        peaks[peaks == 0] = 1.0
        drawdowns = (equity_array - peaks) / peaks
        return float(drawdowns.min())

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
