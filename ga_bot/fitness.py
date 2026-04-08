"""Fitness scoring for the GA — robust, profit-first, win-rate-agnostic.

The user's goal is **live profit with capital preservation**. So this
scorer optimizes the things that actually matter when real money is on
the line:

    1. **Return %** (how much money the strategy makes)
    2. **Calmar** = return / max_dd (return per unit of pain)
    3. **Drawdown safety** — quadratic penalty above 10 % DD
    4. **Single-trade safety** — quadratic penalty when any one trade
       loses more than 3 % of starting capital
    5. **Cross-slice robustness** — train and val must BOTH be profitable
       and must not disagree wildly. The previous version rewarded val
       alone and got gamed by chromosomes that nailed a single regime.

The trick is how train and val are combined. We use:

    base_return = min(train_return, val_return)
    base_dd     = max(train_dd,     val_dd)
    base_worst  = max(train_worst,  val_worst)
    base_pf     = min(train_pf,     val_pf)

i.e. every base metric is the **worse** of the two slices. The GA can
no longer score highly by overfitting to either train or val in
isolation — to climb the fitness curve, both slices have to work.

On top of that, a symmetric divergence penalty crushes any chromosome
where train and val disagree in direction or magnitude. Old one-sided
penalty (`train > val`) missed the opposite case entirely — a
chromosome with train −0.6 % / val +239 % used to score 4.5 flat,
which was exactly the val-overfit collapse the user hit on the Ubuntu
box.

Win rate is *not* in the formula. A 40 % WR strategy with 3:1 R/R and
8 % DD beats a 75 % WR strategy with 35 % DD every single time.

The held-out TEST slice is never touched here. Only `genetic_algorithm`
looks at test, and only for the stop criterion.
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
    train_dd = train_metrics.get("max_dd", 1.0)
    val_dd = val_metrics.get("max_dd", 1.0)
    train_pf = train_metrics.get("profit_factor", 0.0)
    val_pf = val_metrics.get("profit_factor", 0.0)
    train_worst = train_metrics.get("worst_trade_pct", 1.0)
    val_worst = val_metrics.get("worst_trade_pct", 1.0)

    # Activity floors. Both slices must produce a meaningful trade count
    # before we trust any of their metrics.
    activity_train = min(1.0, n_t / float(min_trades_train))
    activity_val = min(1.0, n_v / float(min_trades_val))
    activity = activity_train * activity_val

    # ----- robust (worse-of-both) aggregation -----
    base_ret = min(train_ret, val_ret)
    base_dd = max(train_dd, val_dd)
    base_pf = min(train_pf, val_pf)
    base_worst = max(train_worst, val_worst)

    # ----- refuse losing strategies, but keep a gradient -----
    # BOTH train and val must be profitable. If either slice is in the
    # red, drop into a fallback branch that still gives the GA a
    # gradient toward zero but can never outscore a real winner.
    if base_ret <= 0.0 or base_pf < 1.0:
        return float(-3.0 + max(base_ret, -1.0) * 2.0)

    # ----- profit-first base reward -----
    # tanh keeps the curve smooth and bounded. Both terms use the WORSE
    # slice so the GA can't cherry-pick one.
    profit_term = float(np.tanh(base_ret * 2.0))        # ~0.76 at +50%, saturates near +100%
    base_calmar = base_ret / max(base_dd, 1e-3)
    calmar_term = float(np.tanh(base_calmar / 3.0))     # rewards return-per-DD; saturates ~Calmar=6

    # ----- safety penalties -----
    # Drawdown: free up to 10 %, quadratic above. Uses the DEEPER of
    # train and val DDs so a huge train drawdown can't be hidden by a
    # tame val drawdown.
    dd_excess = max(0.0, base_dd - 0.10)
    dd_penalty = (dd_excess * dd_excess) * 10.0

    # Worst single trade: free up to 3 % of starting capital, quadratic
    # above. Uses the LARGER of the two slices' worst trades.
    worst_excess = max(0.0, base_worst - 0.03)
    worst_penalty = (worst_excess * worst_excess) * 20.0

    # ----- symmetric divergence penalty -----
    # Train and val should agree in magnitude, not just sign. Penalize
    # |train_ret - val_ret| regardless of direction. A chromosome with
    # train 10 % / val 12 % is fine (gap 0.02 → penalty 0.0002). One
    # with train -0.6 % / val +240 % is not (gap 2.4 → penalty 2.88).
    gap = abs(train_ret - val_ret)
    divergence_penalty = gap * gap * 0.5

    base = 2.0 * profit_term + 2.5 * calmar_term
    return float(base * activity - dd_penalty - worst_penalty - divergence_penalty)


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
