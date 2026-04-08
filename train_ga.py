"""Train the GA trading bot on historical XAUUSD M5 data.

Usage:
    python train_ga.py --csv XAUUSD_M5.csv

Stops automatically the moment the best chromosome's *validation* split
satisfies all of:
    win_rate         >= GA.target_win_rate          (default 0.80)
    trades           >= GA.min_trades_for_stop      (default 50)
    profit_factor    >= GA.min_profit_factor_for_stop (default 1.20)

The winning chromosome is saved to ga_bot/models/best_chromosome.json
and is then loadable by run_paper.py / run_live.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ga_bot.config import CONFIG, LOGS_DIR, MODELS_DIR
from ga_bot.data_loader import load_csv, split_train_val
from ga_bot.genetic_algorithm import GeneticAlgorithm, GenerationLog

TRAINING_LOG_PATH = LOGS_DIR / "training.jsonl"


def _print_progress(log: GenerationLog) -> None:
    tm = log.best_train_metrics
    vm = log.best_val_metrics
    print(
        f"gen {log.generation:03d} | "
        f"fit={log.best_fitness:7.3f} | "
        f"train wr={tm['win_rate']*100:5.1f}% pf={tm['profit_factor']:5.2f} "
        f"n={tm['trades']:4d} | "
        f"val   wr={vm['win_rate']*100:5.1f}% pf={vm['profit_factor']:5.2f} "
        f"n={vm['trades']:4d} | "
        f"{log.elapsed_sec:5.1f}s",
        flush=True,
    )
    # Append a structured row the dashboard can read.
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "generation": log.generation,
        "fitness": float(log.best_fitness),
        "train": {k: float(v) for k, v in tm.items()},
        "val": {k: float(v) for k, v in vm.items()},
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
    parser.add_argument("--target-win-rate", type=float, default=None)
    args = parser.parse_args()

    if args.generations is not None:
        CONFIG.ga.max_generations = args.generations
    if args.population is not None:
        CONFIG.ga.population_size = args.population
    if args.target_win_rate is not None:
        CONFIG.ga.target_win_rate = args.target_win_rate

    # Reset training log so the dashboard reflects this run, not previous ones.
    if TRAINING_LOG_PATH.exists():
        TRAINING_LOG_PATH.unlink()
    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_LOG_PATH.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "start",
            "csv": args.csv,
            "target_win_rate": CONFIG.ga.target_win_rate,
            "min_trades_for_stop": CONFIG.ga.min_trades_for_stop,
            "min_profit_factor_for_stop": CONFIG.ga.min_profit_factor_for_stop,
            "max_generations": CONFIG.ga.max_generations,
            "population_size": CONFIG.ga.population_size,
        }) + "\n")

    print(f"Loading {args.csv} ...")
    df = load_csv(args.csv)
    train_df, val_df = split_train_val(df)
    print(
        f"Loaded {len(df):,} bars  |  train={len(train_df):,}  val={len(val_df):,}\n"
        f"Symbol={CONFIG.instrument.symbol}  TF={CONFIG.trading.timeframe_minutes}m  "
        f"Leverage=1:{CONFIG.account.leverage}  Start=${CONFIG.account.starting_balance}\n"
        f"Stop criterion: val WR >= {CONFIG.ga.target_win_rate*100:.0f}% "
        f"AND trades >= {CONFIG.ga.min_trades_for_stop} "
        f"AND PF >= {CONFIG.ga.min_profit_factor_for_stop}"
    )

    ga = GeneticAlgorithm(train_df=train_df, val_df=val_df, on_generation=_print_progress)
    best = ga.run(save_path=Path(args.out))

    val_wr = best.metrics.get("val_win_rate", 0.0)
    val_n = int(best.metrics.get("val_trades", 0))
    val_pf = best.metrics.get("val_profit_factor", 0.0)

    print("\n=== TRAINING DONE ===")
    print(f"Saved to: {args.out}")
    print(f"Validation: win_rate={val_wr*100:.2f}%  trades={val_n}  profit_factor={val_pf:.2f}")

    target_hit = val_wr >= CONFIG.ga.target_win_rate and val_n >= CONFIG.ga.min_trades_for_stop
    with TRAINING_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "done",
            "target_hit": bool(target_hit),
            "saved_to": args.out,
            "val_win_rate": val_wr,
            "val_trades": val_n,
            "val_profit_factor": val_pf,
        }) + "\n")

    if target_hit:
        print("Target win rate reached. Ready for paper trading.")
        return 0
    print("Target win rate NOT reached — best-effort model saved. Re-run with more generations or different data.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
