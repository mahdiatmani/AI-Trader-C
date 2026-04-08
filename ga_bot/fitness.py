"""Fitness scoring for the GA — profit-first, win-rate-agnostic.

The user's goal is **live profit with capital preservation**, not a
pretty win-rate number. So this scorer optimizes the things that
actually matter when real money is on the line:

    1. Validation **return %** (how much money the strategy makes)
    2. Validation **Calmar** = return / max_dd (return per unit of pain)
    3. **Drawdown safety** — quadratic penalty above 10 % DD
    4. **Single-trade safety** — quadratic penalty when any one trade
       loses more than 3 % of starting capital
    5. **Overfit penalty** — train return must not be wildly larger than
       val return, otherwise the GA is just memorizing the train slice

Win rate is *not* in the formula. A 40 % WR strategy with 3:1 R/R and
8 % DD beats a 75 % WR strategy with 35 % DD every single time, and the
old WR-led fitness used to throw the first one away. We don't anymore.

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

    train_ret = train_metrics.get("return_pct", 0.0)
    val_ret = val_metrics.get("return_pct", 0.0)
    val_dd = val_metrics.get("max_dd", 1.0)
    val_pf = val_metrics.get("profit_factor", 0.0)
    val_calmar = val_metrics.get("calmar", 0.0)
    val_worst = val_metrics.get("worst_trade_pct", 1.0)

    # Activity floors. Both sides must produce a meaningful trade count
    # before we trust the metrics. Train is the bigger constraint —
    # 80 trades over the train slice keeps the GA from cherry-picking a
    # handful of lucky setups.
    activity_train = min(1.0, n_t / float(min_trades_train))
    activity_val = min(1.0, n_v / float(min_trades_val))
    activity = activity_train * activity_val

    # ----- refuse losing strategies, but keep a gradient -----
    # If validation is in the red we still want the GA to climb toward
    # zero, but the ceiling is hard so a losing chromosome can never
    # outscore a profitable one.
    if val_ret <= 0.0 or val_pf < 1.0:
        return float(-3.0 + max(val_ret, -1.0) * 2.0)

    # ----- profit-first base reward -----
    # tanh keeps the curve smooth and bounded so a single absurd outlier
    # can't dominate selection.
    profit_term = float(np.tanh(val_ret * 2.0))            # ~0.76 at +50%, saturates near +100%
    calmar_term = float(np.tanh(val_calmar / 3.0))         # rewards return-per-DD; saturates ~Calmar=6

    # ----- safety penalties -----
    # Drawdown: free up to 10 %, quadratic above. A 20 % DD costs 0.10,
    # a 30 % DD costs 0.40 — i.e. shallow DDs are forgiven, deep ones get
    # crushed.
    dd_excess = max(0.0, val_dd - 0.10)
    dd_penalty = (dd_excess * dd_excess) * 10.0

    # Worst single trade: free up to 3 % of starting capital, quadratic
    # above. The user explicitly does not want a strategy that nukes the
    # account on one bad fill.
    worst_excess = max(0.0, val_worst - 0.03)
    worst_penalty = (worst_excess * worst_excess) * 20.0

    # ----- overfit penalty -----
    # If train return is much bigger than val return, the chromosome is
    # memorizing the train slice. We allow train to be modestly higher
    # (up to 1.5x val return) for free, then crush divergence.
    ret_gap = max(0.0, train_ret - max(val_ret, 0.0) * 1.5)
    overfit_penalty = ret_gap * ret_gap * 0.5

    base = 2.0 * profit_term + 2.5 * calmar_term
    return float(base * activity - dd_penalty - worst_penalty - overfit_penalty)


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
