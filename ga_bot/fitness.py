"""Fitness scoring for the GA.

A chromosome's fitness is a single scalar that the GA maximizes. This
revision uses BOTH the train and the validation slice so the GA is
actively penalized for overfitting:

    * the base reward comes from VAL metrics (the half the GA can see
      but is not allowed to memorize cheaply)
    * a quadratic penalty kicks in when train_wr is much higher than
      val_wr — that's the textbook overfit signature
    * a hard activity floor on TRAIN trade count refuses chromosomes
      that won big on a handful of cherry-picked setups (the
      lucky-30-trades trap)

The held-out TEST slice is never touched here. Only `genetic_algorithm`
looks at the test set, and only for the stop criterion.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .backtester import BacktestResult


def score(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    min_trades_train: int = 80,
    min_trades_val: int = 20,
) -> float:
    n_t = train_metrics.get("trades", 0)
    n_v = val_metrics.get("trades", 0)
    if n_t < 5 or n_v < 5:
        return -10.0  # essentially never selected

    train_wr = train_metrics.get("win_rate", 0.0)
    val_wr = val_metrics.get("win_rate", 0.0)
    val_pf = val_metrics.get("profit_factor", 0.0)
    val_net = val_metrics.get("net_pnl", 0.0)
    val_dd = val_metrics.get("max_dd", 1.0)

    # Activity floors. Both sides must produce a meaningful trade count
    # before we trust the metrics. Train is the bigger constraint —
    # 80 trades over the train slice is what kills the lucky-30-trades
    # chromosomes that the previous fitness happily rewarded.
    activity_train = min(1.0, n_t / float(min_trades_train))
    activity_val = min(1.0, n_v / float(min_trades_val))
    activity = activity_train * activity_val

    # Overfit penalty: quadratic on the train-vs-val win-rate gap.
    # A 5 % gap costs ~0.0125, a 30 % gap costs 0.45 — i.e. small gaps
    # are forgiven, big gaps are punished hard.
    gap = max(0.0, train_wr - val_wr)
    overfit_penalty = (gap * gap) * 5.0

    # Base reward — driven by VALIDATION, not training.
    if val_net <= 0 or val_pf < 1.0:
        # Still let the GA gradient toward higher val WR even when
        # validation is losing money, but cap the ceiling so unprofitable
        # chromosomes can never compete with profitable ones.
        base = -3.0 + (val_wr - 0.5) * 2.0
    else:
        wr_term = (val_wr - 0.5) * 2.0          # -1 .. +1
        pf_term = float(np.tanh(val_pf - 1.0))  # 0 at PF=1, saturates near 1
        dd_term = -val_dd
        base = (
            2.0 * wr_term
            + 1.5 * pf_term
            + 0.5 * dd_term
            + 0.3 * float(np.log1p(max(val_net, 0.0))) / 5.0
        )

    return float(base * activity - overfit_penalty)


def evaluate(
    train_result: BacktestResult,
    val_result: BacktestResult,
    min_trades_train: int = 80,
    min_trades_val: int = 20,
) -> Dict[str, float]:
    """Compose a single metrics dict for a chromosome.

    The returned dict carries both train_* and val_* keys plus a single
    `fitness` scalar so the GA can sort populations cheaply.
    """
    tm = train_result.metrics()
    vm = val_result.metrics()
    fit = score(
        tm, vm,
        min_trades_train=min_trades_train,
        min_trades_val=min_trades_val,
    )
    out: Dict[str, float] = {f"train_{k}": float(v) for k, v in tm.items()}
    out.update({f"val_{k}": float(v) for k, v in vm.items()})
    # Keep the bare keys for backward compat with the printer in
    # train_ga.py and the dashboard's training panel.
    out.update({k: float(v) for k, v in tm.items()})
    out["fitness"] = fit
    return out
