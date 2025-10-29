"""Risk analytics helpers for reporting modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from risk.position_sizing import PositionSizingResult


@dataclass(frozen=True)
class RiskMetrics:
    """Bundle of risk related metrics for reporting."""

    value_at_risk: float
    capital_at_risk: float


def historical_var(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Compute a simple historical Value at Risk (VaR).

    The function expects a sequence of returns expressed as decimals
    (e.g. ``0.01`` = 1%). VaR is returned as a positive number
    representing the potential loss for the given confidence level.
    """

    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    if len(returns) == 0:
        raise ValueError("Returns sequence must not be empty.")

    sorted_returns = sorted(returns)
    index = max(int((1 - confidence) * len(sorted_returns)) - 1, 0)
    var = sorted_returns[index]
    return abs(min(var, 0.0))


def capital_at_risk(positions: Iterable[PositionSizingResult]) -> float:
    """Aggregate the risk amount for the provided positions."""

    return float(sum(max(result.risk_amount, 0.0) for result in positions))


def build_risk_metrics(
    returns: Sequence[float],
    positions: Iterable[PositionSizingResult],
    confidence: float = 0.95,
) -> RiskMetrics:
    """Construct a :class:`RiskMetrics` object combining VaR and CAR."""

    return RiskMetrics(
        value_at_risk=historical_var(returns, confidence=confidence),
        capital_at_risk=capital_at_risk(positions),
    )


def trade_distribution_metrics(r_values: Sequence[float]) -> dict[str, float]:
    """Compute expectancy-focused statistics from a sequence of R multiples."""

    r_array = np.array(list(r_values), dtype=float)
    if r_array.size == 0:
        return {
            "Average Win (R)": 0.0,
            "Average Loss (R)": 0.0,
            "RR Effective": 0.0,
            "Breakeven Winrate %": 0.0,
            "Expectancy (R)": 0.0,
            "Winrate": 0.0,
        }

    wins = r_array[r_array > 0]
    losses = r_array[r_array < 0]

    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0

    winrate = float(wins.size) / float(r_array.size)
    rr_effective = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")
    breakeven_winrate = (
        abs(avg_loss) / (abs(avg_loss) + avg_win) if (avg_win + abs(avg_loss)) > 0 else 0.0
    )
    expectancy = winrate * avg_win + (1 - winrate) * avg_loss

    return {
        "Average Win (R)": avg_win,
        "Average Loss (R)": avg_loss,
        "RR Effective": rr_effective if np.isfinite(rr_effective) else float("inf"),
        "Breakeven Winrate %": breakeven_winrate * 100,
        "Expectancy (R)": expectancy,
        "Winrate": winrate,
    }


def prepare_drawdown_columns(df_trades: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df_trades`` enriched with peak and drawdown columns."""

    if "capital_final" not in df_trades.columns:
        raise ValueError("'capital_final' column is required to compute drawdown.")

    enriched = df_trades.copy()
    enriched["peak"] = enriched["capital_final"].cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["drawdown"] = (
            (enriched["capital_final"] - enriched["peak"]) / enriched["peak"]
        ) * 100
    enriched["drawdown"] = enriched["drawdown"].fillna(0.0)
    return enriched


def compute_backtest_summary_metrics(
    trades_df: pd.DataFrame, summary_df: pd.DataFrame | Mapping[str, float] | None = None
) -> dict[str, float]:
    """Aggregate the core metrics required for the analytics dashboard."""

    if trades_df.empty:
        return {
            "Winrate %": 0.0,
            "Net Profit %": 0.0,
            "Expectancy (R)": 0.0,
            "Average Win (R)": 0.0,
            "Average Loss (R)": 0.0,
            "RR Effective": 0.0,
            "Breakeven Winrate %": 0.0,
            "Max Drawdown %": 0.0,
            "Profit Factor": 0.0,
        }

    metrics = trade_distribution_metrics(trades_df["r_multiple"].fillna(0.0))

    winrate_pct = metrics["Winrate"] * 100
    expectancy = metrics["Expectancy (R)"]
    avg_win = metrics["Average Win (R)"]
    avg_loss = metrics["Average Loss (R)"]
    rr_effective = metrics["RR Effective"] if np.isfinite(metrics["RR Effective"]) else 0.0
    breakeven = metrics["Breakeven Winrate %"]

    capital_series = trades_df["capital_final"].astype(float)
    initial_capital = capital_series.iloc[0]
    final_capital = capital_series.iloc[-1]
    net_profit_pct = ((final_capital / initial_capital) - 1.0) * 100 if initial_capital else 0.0

    enriched = prepare_drawdown_columns(trades_df)
    max_drawdown_pct = enriched["drawdown"].min()

    positive_r = trades_df.loc[trades_df["r_multiple"] > 0, "r_multiple"]
    negative_r = trades_df.loc[trades_df["r_multiple"] < 0, "r_multiple"]
    total_wins = float(positive_r.sum())
    total_losses = float(negative_r.sum())
    profit_factor = abs(total_wins) / abs(total_losses) if total_losses != 0 else float("inf")
    profit_factor = profit_factor if np.isfinite(profit_factor) else float("inf")

    summary_metrics = {
        "Winrate %": winrate_pct,
        "Net Profit %": net_profit_pct,
        "Expectancy (R)": expectancy,
        "Average Win (R)": avg_win,
        "Average Loss (R)": avg_loss,
        "RR Effective": rr_effective,
        "Breakeven Winrate %": breakeven,
        "Max Drawdown %": abs(max_drawdown_pct),
        "Profit Factor": profit_factor,
    }

    if summary_df is not None:
        if isinstance(summary_df, pd.DataFrame):
            row = summary_df.iloc[0]
            for key in summary_metrics.keys():
                if key in row:
                    summary_metrics[key] = float(row[key])
        else:
            for key in summary_metrics.keys():
                if key in summary_df:
                    summary_metrics[key] = float(summary_df[key])

    return summary_metrics


__all__ = [
    "RiskMetrics",
    "historical_var",
    "capital_at_risk",
    "build_risk_metrics",
    "trade_distribution_metrics",
    "prepare_drawdown_columns",
    "compute_backtest_summary_metrics",
]
