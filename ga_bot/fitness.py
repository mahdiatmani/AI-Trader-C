"""Fitness scoring for the GA.

A chromosome's fitness is a single scalar that the GA maximizes. We blend
multiple metrics so the search rewards strategies that are simultaneously:
    * accurate (high win rate)
    * profitable (positive net P&L, profit factor > 1)
    * active (enough trades — not 3 lucky ones)
    * stable (low max drawdown)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .backtester import BacktestResult


def score(metrics: Dict[str, float], min_trades: int = 30) -> float:
    n = metrics["trades"]
    if n < 5:
        return -10.0  # essentially never selected

    win_rate = metrics["win_rate"]
    pf = metrics["profit_factor"]
    net = metrics["net_pnl"]
    dd = metrics["max_dd"]

    # Penalize unprofitable strategies regardless of WR
    if net <= 0:
        return -5.0 + win_rate  # let GA still gradient toward higher WR

    # Trade-count ramp: don't fully trust scores until we have enough samples
    activity = min(1.0, n / float(min_trades))

    pf_term = np.tanh((pf - 1.0))           # ~0 at PF=1, saturates near 1
    wr_term = (win_rate - 0.5) * 2.0         # -1 .. +1
    dd_term = -dd                            # less drawdown is better

    s = (
        2.0 * wr_term
        + 1.5 * pf_term
        + 0.5 * dd_term
        + 0.5 * np.log1p(max(net, 0.0)) / 5.0
    ) * activity
    return float(s)


def evaluate(result: BacktestResult, min_trades: int = 30) -> Dict[str, float]:
    m = result.metrics()
    m["fitness"] = score(m, min_trades=min_trades)
    return m
