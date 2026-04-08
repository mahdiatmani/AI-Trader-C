"""Phase 2: live trading via MetaApi (real money).

DO NOT run this until:
    1. The chromosome has met the win-rate stop criterion in training, AND
    2. It has been paper-traded against a live feed for a meaningful sample.

This script intentionally requires an explicit `--i-understand-the-risk`
flag so it cannot be launched by accident.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ga_bot.broker.metaapi_live import MetaApiLiveBroker
from ga_bot.chromosome import Chromosome
from ga_bot.config import CONFIG, MODELS_DIR
from ga_bot.trader import Trader, parse_duration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODELS_DIR / "best_chromosome.json"))
    parser.add_argument("--i-understand-the-risk", action="store_true")
    parser.add_argument(
        "--duration",
        default=None,
        help="Auto-stop after this elapsed time, e.g. 12h, 7d, 2w. Omit to run forever.",
    )
    args = parser.parse_args()

    if not args.i_understand_the_risk:
        print(
            "Refusing to start live trading. Re-run with "
            "--i-understand-the-risk if you really mean it."
        )
        return 2

    chromosome = Chromosome.load(Path(args.model))
    print(f"Loaded chromosome fitness={chromosome.fitness:.3f}")
    print(f"Validation metrics: {chromosome.metrics}")
    print(
        f"LIVE on {CONFIG.instrument.symbol} TF={CONFIG.trading.timeframe_minutes}m "
        f"Leverage=1:{CONFIG.account.leverage}"
    )

    broker = MetaApiLiveBroker()
    max_runtime = parse_duration(args.duration) if args.duration else 0.0
    trader = Trader(broker=broker, chromosome=chromosome, max_runtime_sec=max_runtime)
    if max_runtime:
        print(f"Live trading for {args.duration} (Ctrl+C / SIGTERM to stop early).")
    trader.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
