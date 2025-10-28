"""Simple candlestick pattern recognition strategy."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import BaseStrategy, SignalDataFrame


class CandlestickPatternStrategy(BaseStrategy):
    """Detect basic bullish or bearish candlestick reversals."""

    default_params: Dict[str, float] = {
        "pattern": "bullish_engulfing",
        "support_window": 20,
        "resistance_window": 20,
        "volume_lookback": 10,
        "volume_multiplier": 1.0,
        "ema_period": 50,
        "proximity_pct": 0.003,
    }

    def generate_signals(self, data: pd.DataFrame) -> SignalDataFrame:
        """Generate engulfing pattern signals for the provided OHLC data."""

        frame = data.copy()
        frame = frame.sort_index()
        signals = pd.Series(0, index=frame.index, dtype=int)

        if len(frame) < 2:
            return pd.DataFrame({"signal": signals})

        previous = frame.shift(1)
        pattern = str(self.params.get("pattern", "bullish_engulfing")).lower()
        volume_lookback = int(self.params.get("volume_lookback", 10))
        volume_multiplier = float(self.params.get("volume_multiplier", 1.0))
        ema_period = int(self.params.get("ema_period", 50))
        support_window = int(self.params.get("support_window", 20))
        resistance_window = int(self.params.get("resistance_window", 20))
        proximity_pct = float(self.params.get("proximity_pct", 0.003))

        open_prices = frame.get("open", frame["close"])
        prev_open = previous.get("open", previous["close"])

        ema = frame["close"].ewm(span=ema_period, adjust=False).mean()
        default_low = frame[[col for col in ("open", "close") if col in frame]].min(axis=1)
        default_high = frame[[col for col in ("open", "close") if col in frame]].max(axis=1)
        low = frame.get("low", default_low)
        high = frame.get("high", default_high)
        volume_series = frame.get("volume", pd.Series(1, index=frame.index))
        avg_volume = volume_series.rolling(window=volume_lookback, min_periods=1).mean()
        current_volume = volume_series
        volume_ok = current_volume >= avg_volume * volume_multiplier

        recent_low = low.rolling(window=support_window, min_periods=1).min()
        recent_high = high.rolling(window=resistance_window, min_periods=1).max()
        insufficient_history = len(frame) < max(support_window, resistance_window)
        bullish_zone = (frame["close"] <= recent_low * (1 + proximity_pct)) | insufficient_history
        bearish_zone = (frame["close"] >= recent_high * (1 - proximity_pct)) | insufficient_history

        if pattern == "bullish_engulfing":
            prev_bearish = previous["close"] < prev_open
            curr_bullish = frame["close"] > open_prices
            body_engulf = (frame["close"] >= prev_open) & (
                open_prices <= previous["close"]
            )
            trend_ok = (ema > ema.shift(1)) | insufficient_history
            bullish = (
                prev_bearish
                & curr_bullish
                & body_engulf
                & bullish_zone
                & trend_ok
                & volume_ok
            )
            signals.loc[bullish.fillna(False)] = 1
        elif pattern == "bearish_engulfing":
            prev_bullish = previous["close"] > prev_open
            curr_bearish = frame["close"] < open_prices
            body_engulf = (frame["close"] <= prev_open) & (
                open_prices >= previous["close"]
            )
            trend_ok = (ema < ema.shift(1)) | insufficient_history
            bearish = (
                prev_bullish
                & curr_bearish
                & body_engulf
                & bearish_zone
                & trend_ok
                & volume_ok
            )
            signals.loc[bearish.fillna(False)] = -1

        return pd.DataFrame({"signal": signals})
