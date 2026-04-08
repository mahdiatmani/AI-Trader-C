"""Train the GA trading bot on historical XAUUSD M5 data.

Usage:
    python train_ga.py --csv XAUUSD_M5.csv

Stops automatically the moment the best chromosome's **held-out test
slice** satisfies ALL of:
    return_pct       >= GA.target_return_pct          (default +50 %)
    max_dd           <= GA.max_acceptable_dd          (default 20 %)
    worst_trade_pct  <= GA.max_worst_trade_pct        (default 5 %)
    profit_factor    >= GA.min_profit_factor_for_stop (default 1.50)
    trades           >= GA.min_trades_for_stop        (default 50)

Win rate is intentionally not in the stop criteria — the goal is live
profit with capital preservation, not a pretty WR number.

The winning chromosome is saved to ga_bot/models/best_chromosome.json
and is then loadable by run_paper.py / run_live.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ga_bot.config import CONFIG, LOGS_DIR, MODELS_DIR
from ga_bot.data_loader import load_csv, split_train_folds_val_test
from ga_bot.genetic_algorithm import GeneticAlgorithm, GenerationLog

TRAINING_LOG_PATH = LOGS_DIR / "training.jsonl"


def _fmt_split(label: str, m: dict) -> str:
    start = float(CONFIG.account.starting_balance)
    end_balance = start * (1.0 + m.get("return_pct", 0.0))
    return (
        f"{label} bal=${end_balance:8.2f} "
        f"ret={m.get('return_pct', 0.0)*100:+6.1f}% "
        f"dd={m.get('max_dd', 0.0)*100:5.1f}% "
        f"worst={m.get('worst_trade_pct', 0.0)*100:4.1f}% "
        f"pf={m.get('profit_factor', 0.0):4.2f} "
        f"n={int(m.get('trades', 0)):4d}"
    )


def _print_progress(log: GenerationLog) -> None:
    tm = log.best_train_metrics
    vm = log.best_val_metrics
    em = log.best_test_metrics
    print(
        f"gen {log.generation:03d} | "
        f"fit={log.best_fitness:7.3f} | "
        f"{_fmt_split('train', tm)} | "
        f"{_fmt_split('val',   vm)} | "
        f"{_fmt_split('test',  em)} | "
        f"{log.elapsed_sec:5.1f}s",
        flush=True,
    )
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "generation": log.generation,
        "fitness": float(log.best_fitness),
        "train": {k: float(v) for k, v in tm.items()},
        "val": {k: float(v) for k, v in vm.items()},
        "test": {k: float(v) for k, v in em.items()},
        "elapsed_sec": float(log.elapsed_sec),
    }
    with TRAINING_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train GA trading bot on XAUUSD M5.")
    parser.add_argument("--csv", required=True, help="CSV file in ./data/ (or absolute path)")
    parser.add_argument("--out", default=str(MODELS_DIR / "best_chromosome.json"))
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument(
        "--target-return", type=float, default=None,
        help="Target return on TEST slice as a fraction (e.g. 0.50 = +50%%)",
    )
    parser.add_argument(
        "--max-dd", type=float, default=None,
        help="Max acceptable drawdown on TEST slice as a fraction (e.g. 0.20 = 20%%)",
    )
    parser.add_argument(
        "--max-worst-trade", type=float, default=None,
        help="Max acceptable single-trade loss as a fraction of starting capital",
    )
    args = parser.parse_args()

    if args.generations is not None:
        CONFIG.ga.max_generations = args.generations
    if args.population is not None:
        CONFIG.ga.population_size = args.population
    if args.target_return is not None:
        CONFIG.ga.target_return_pct = args.target_return
    if args.max_dd is not None:
        CONFIG.ga.max_acceptable_dd = args.max_dd
    if args.max_worst_trade is not None:
        CONFIG.ga.max_worst_trade_pct = args.max_worst_trade

    # Reset training log so the dashboard reflects this run, not previous ones.
    if TRAINING_LOG_PATH.exists():
        TRAINING_LOG_PATH.unlink()
    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_LOG_PATH.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "start",
            "csv": args.csv,
            "target_return_pct": CONFIG.ga.target_return_pct,
            "max_acceptable_dd": CONFIG.ga.max_acceptable_dd,
            "max_worst_trade_pct": CONFIG.ga.max_worst_trade_pct,
            "min_profit_factor_for_stop": CONFIG.ga.min_profit_factor_for_stop,
            "min_trades_for_stop": CONFIG.ga.min_trades_for_stop,
            "max_generations": CONFIG.ga.max_generations,
            "population_size": CONFIG.ga.population_size,
        }) + "\n")

    print(f"Loading {args.csv} ...")
    df = load_csv(args.csv)
    train_folds, val_df, test_df = split_train_folds_val_test(df)
    fold_sizes = " ".join(f"f{i}={len(f):,}" for i, f in enumerate(train_folds))
    train_total = sum(len(f) for f in train_folds)
    print(
        f"Loaded {len(df):,} bars  |  train={train_total:,} ({len(train_folds)} folds: {fold_sizes})  "
        f"val={len(val_df):,}  test={len(test_df):,}\n"
        f"Symbol={CONFIG.instrument.symbol}  TF={CONFIG.trading.timeframe_minutes}m  "
        f"Leverage=1:{CONFIG.account.leverage}  Start=${CONFIG.account.starting_balance}\n"
        f"Fitness rule: every train fold AND val must be profitable; base "
        f"metrics use the WORST slice across all {len(train_folds)+1}.\n"
        f"Stop criterion (TEST set, fully held out): "
        f"return >= {CONFIG.ga.target_return_pct*100:+.0f}% "
        f"AND DD <= {CONFIG.ga.max_acceptable_dd*100:.0f}% "
        f"AND worst trade <= {CONFIG.ga.max_worst_trade_pct*100:.0f}% "
        f"AND PF >= {CONFIG.ga.min_profit_factor_for_stop:.2f} "
        f"AND trades >= {CONFIG.ga.min_trades_for_stop}"
    )

    ga = GeneticAlgorithm(
        train_folds=train_folds,
        val_df=val_df,
        test_df=test_df,
        on_generation=_print_progress,
    )
    best = ga.run(save_path=Path(args.out))

    test_ret = best.metrics.get("test_return_pct", 0.0)
    test_dd = best.metrics.get("test_max_dd", 0.0)
    test_worst = best.metrics.get("test_worst_trade_pct", 0.0)
    test_pf = best.metrics.get("test_profit_factor", 0.0)
    test_n = int(best.metrics.get("test_trades", 0))

    start_bal = float(CONFIG.account.starting_balance)
    test_bal = start_bal * (1.0 + test_ret)

    print("\n=== TRAINING DONE ===")
    print(f"Saved to: {args.out}")
    print(
        f"Held-out TEST: start=${start_bal:.2f} -> end=${test_bal:.2f}  "
        f"({test_ret*100:+.2f}%)  "
        f"max_dd={test_dd*100:.2f}%  "
        f"worst_trade={test_worst*100:.2f}%  "
        f"PF={test_pf:.2f}  "
        f"trades={test_n}"
    )

    target_hit = (
        test_n >= CONFIG.ga.min_trades_for_stop
        and test_ret >= CONFIG.ga.target_return_pct
        and test_dd <= CONFIG.ga.max_acceptable_dd
        and test_worst <= CONFIG.ga.max_worst_trade_pct
        and test_pf >= CONFIG.ga.min_profit_factor_for_stop
    )
    with TRAINING_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "done",
            "target_hit": bool(target_hit),
            "saved_to": args.out,
            "test_return_pct": test_ret,
            "test_max_dd": test_dd,
            "test_worst_trade_pct": test_worst,
            "test_profit_factor": test_pf,
            "test_trades": test_n,
        }) + "\n")

    if target_hit:
        print("Target reached on held-out test set. Ready for paper trading.")
        return 0
    print("Target NOT reached on held-out test set — best-effort model saved. Re-run with more generations or different data.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
