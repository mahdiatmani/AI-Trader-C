"""Fitness scoring for the GA — robust, profit-first, win-rate-agnostic.

The user's goal is **live profit with capital preservation**. So this
scorer optimizes the things that actually matter when real money is on
the line:

    1. **Return %** (how much money the strategy makes)
    2. **Calmar** = return / max_dd (return per unit of pain)
    3. **Drawdown safety** — quadratic penalty above 10 % DD
    4. **Single-trade safety** — quadratic penalty when any one trade
       loses more than 3 % of starting capital
    5. **Multi-regime robustness** — train is chopped into N contiguous
       sub-folds. Every base metric is the **WORST** value across all
       folds + val, so a strategy that only works in one regime can't
       win. Single-slice fitness gets gamed every time; multi-slice
       worst-of-N is the structural fix.

Aggregation rule for every base metric:

    base_return = min(fold_return for all folds) ∪ val_return
    base_dd     = max(fold_dd     for all folds) ∪ val_dd
    base_pf     = min(fold_pf     for all folds) ∪ val_pf
    base_worst  = max(fold_worst  for all folds) ∪ val_worst

i.e. each base metric is the **worst** value across N+1 slices. The GA
can only score highly if the chromosome survives every regime.

On top of that, two divergence penalties crush spread:
    - return divergence: (max_ret - min_ret)² across all slices
    - DD divergence: (max_dd - min_dd)² across all slices

Tanh saturation knees were pulled out (knee at +250 % return / Calmar
12) so the GA never locks on a "decent" plateau the way it did when the
knees were at +50 % / Calmar 6 — that's what froze gen 89→153 at 4.4755.

Win rate is *not* in the formula. A 40 % WR strategy with 3:1 R/R and
8 % DD beats a 75 % WR strategy with 35 % DD every single time.

The held-out TEST slice is never touched here. Only `genetic_algorithm`
looks at test, and only for the stop criterion.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .backtester import BacktestResult


def score(
    train_fold_metrics: Sequence[Dict[str, float]],
    val_metrics: Dict[str, float],
    min_trades_per_fold: int = 20,
    min_trades_val: int = 20,
) -> float:
    if not train_fold_metrics:
        return -10.0

    # ----- activity floor: every fold AND val must trade enough -----
    fold_trades = [m.get("trades", 0) for m in train_fold_metrics]
    n_v = val_metrics.get("trades", 0)
    if any(n < 5 for n in fold_trades) or n_v < 5:
        return -10.0

    activities = [min(1.0, n / float(min_trades_per_fold)) for n in fold_trades]
    activities.append(min(1.0, n_v / float(min_trades_val)))
    activity = min(activities)  # the bottleneck slice sets the ceiling

    # ----- gather every slice's metrics -----
    all_metrics: List[Dict[str, float]] = list(train_fold_metrics) + [val_metrics]
    rets = [m.get("return_pct", 0.0) for m in all_metrics]
    dds = [m.get("max_dd", 1.0) for m in all_metrics]
    pfs = [m.get("profit_factor", 0.0) for m in all_metrics]
    worsts = [m.get("worst_trade_pct", 1.0) for m in all_metrics]

    # ----- robust (worst-of-all) aggregation -----
    base_ret = min(rets)
    base_dd = max(dds)
    base_pf = min(pfs)
    base_worst = max(worsts)

    # ----- refuse losing strategies, but keep a gradient -----
    # EVERY slice (all train folds AND val) must be profitable. If even
    # one is in the red, drop into a fallback branch that still gives
    # the GA a gradient toward zero but can never outscore a real
    # winner.
    if base_ret <= 0.0 or base_pf < 1.0:
        return float(-3.0 + max(base_ret, -1.0) * 2.0)

    # ----- profit-first base reward (de-saturated) -----
    # Knees moved well past the previous 4.5 ceiling so the GA can keep
    # differentiating chromosomes when both train folds and val are
    # already healthy. Old: tanh(ret*2.0)+tanh(calmar/3.0) saturated at
    # ret≈+1.0 / calmar≈9. New: knees at ret≈+2.5 / calmar≈12 so a
    # great chromosome can score ~5–6 and an exceptional one ~8+.
    profit_term = float(np.tanh(base_ret * 0.6))         # knee ~+250 %
    base_calmar = base_ret / max(base_dd, 1e-3)
    calmar_term = float(np.tanh(base_calmar / 6.0))      # knee ~Calmar 12

    # ----- safety penalties -----
    # Drawdown: free up to 10 %, quadratic above. Uses the DEEPEST DD
    # across all slices so a huge fold drawdown can't be hidden by a
    # tame val drawdown.
    dd_excess = max(0.0, base_dd - 0.10)
    dd_penalty = (dd_excess * dd_excess) * 10.0

    # Worst single trade: free up to 3 % of starting capital, quadratic
    # above. Uses the LARGEST worst-trade across all slices.
    worst_excess = max(0.0, base_worst - 0.03)
    worst_penalty = (worst_excess * worst_excess) * 20.0

    # ----- divergence penalties -----
    # Return divergence: every slice should produce a similar return.
    # A spread between best and worst slice means the chromosome relies
    # on one regime. Squared so 5 % spread is free-ish (0.0125) but a
    # 50 % spread costs 1.25.
    ret_spread = max(rets) - min(rets)
    ret_divergence_penalty = ret_spread * ret_spread * 0.5

    # DD divergence: same idea. A chromosome with one calm fold (5 %
    # DD) and one wild fold (25 % DD) is fragile even if both made
    # money. 20 % spread → penalty 0.4.
    dd_spread = max(dds) - min(dds)
    dd_divergence_penalty = dd_spread * dd_spread * 10.0

    base = 3.0 * profit_term + 3.0 * calmar_term
    return float(
        base * activity
        - dd_penalty
        - worst_penalty
        - ret_divergence_penalty
        - dd_divergence_penalty
    )


def evaluate(
    train_fold_results: Sequence[BacktestResult],
    val_result: BacktestResult,
    min_trades_per_fold: int = 20,
    min_trades_val: int = 20,
) -> Dict[str, float]:
    """Compose a single metrics dict for a chromosome.

    The returned dict carries:
        - per-fold ``train_fold{i}_*`` keys (one block per fold)
        - ``val_*`` keys for the val slice
        - bare ``train_*`` keys equal to the WORST fold metrics (so the
          per-generation printer shows the bottleneck the GA actually
          optimized against, not an inflated average)
        - a single ``fitness`` scalar
    """
    fold_metrics = [r.metrics() for r in train_fold_results]
    vm = val_result.metrics()

    fit = score(
        fold_metrics, vm,
        min_trades_per_fold=min_trades_per_fold,
        min_trades_val=min_trades_val,
    )

    out: Dict[str, float] = {}
    for i, fm in enumerate(fold_metrics):
        for k, v in fm.items():
            out[f"train_fold{i}_{k}"] = float(v)
    out.update({f"val_{k}": float(v) for k, v in vm.items()})

    # Worst-fold view for the legacy printer keys. Pick the fold with
    # the LOWEST return_pct (matches the bottleneck the fitness is
    # actually scoring against).
    worst_fold_idx = min(
        range(len(fold_metrics)),
        key=lambda i: fold_metrics[i].get("return_pct", 0.0),
    )
    worst_fold = fold_metrics[worst_fold_idx]
    out.update({k: float(v) for k, v in worst_fold.items()})
    out["worst_fold_idx"] = float(worst_fold_idx)
    out["fitness"] = fit
    return out
