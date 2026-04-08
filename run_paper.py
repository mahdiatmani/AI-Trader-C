"""Phase 1: paper trade the saved chromosome.

Two modes:

    --replay <csv>     Replay a CSV historical file bar-by-bar against the
                       PaperBroker. Lets you sanity-check the saved model
                       offline before pointing it at a live feed.

    --live-feed        Use a live MetaApi quote feed (requires
                       metaapi-cloud-sdk + credentials), but route all
                       order execution through the PaperBroker. This is
                       what you run on the Ubuntu cloud instance for true
                       paper trading on real prices.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from ga_bot.broker.paper import PaperBroker
from ga_bot.chromosome import Chromosome
from ga_bot.config import CONFIG, MODELS_DIR
from ga_bot.data_loader import load_csv
from ga_bot.trader import Trader, parse_duration


def _csv_replay_feed(df: pd.DataFrame):
    """Yields a callable that returns the bars up to a moving cursor."""
    cursor = {"i": 200}  # need warmup history

    def _feed(symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        i = cursor["i"]
        start = max(0, i - count)
        return df.iloc[start:i]

    def _advance() -> bool:
        cursor["i"] += 1
        return cursor["i"] <= len(df)

    return _feed, _advance


def _live_feed_factory():
    """Construct a feed callable backed by MetaApi quotes (paper execution)."""
    from ga_bot.broker.metaapi_live import MetaApiLiveBroker

    live = MetaApiLiveBroker()
    live.connect()

    def _feed(symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        return live.get_latest_bars(symbol, timeframe_minutes, count)

    return _feed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODELS_DIR / "best_chromosome.json"))
    parser.add_argument("--replay", help="CSV file to replay (offline paper test)")
    parser.add_argument("--live-feed", action="store_true", help="Use MetaApi quotes for paper execution")
    parser.add_argument(
        "--duration",
        default=None,
        help="Auto-stop after this elapsed time, e.g. 30m, 12h, 7d, 2w (live-feed only)",
    )
    args = parser.parse_args()

    if not (args.replay or args.live_feed):
        parser.error("Provide either --replay <csv> or --live-feed")

    chromosome = Chromosome.load(Path(args.model))
    print(f"Loaded chromosome: fitness={chromosome.fitness:.3f}")
    print(f"Validation metrics: {chromosome.metrics}")

    if args.replay:
        df = load_csv(args.replay)
        feed, advance = _csv_replay_feed(df)
        broker = PaperBroker(bar_feed=feed)
        trader = Trader(broker=broker, chromosome=chromosome)
        broker.connect()
        print(f"Replaying {len(df):,} bars in paper mode ...")
        try:
            while True:
                trader.step()
                if not advance():
                    break
        finally:
            broker.disconnect()
            final_eq = broker.get_equity()
            pnl = final_eq - CONFIG.account.starting_balance
            print(f"Replay done. Final equity = ${final_eq:.2f}  PnL = ${pnl:+.2f}")
        return 0

    feed = _live_feed_factory()
    broker = PaperBroker(bar_feed=feed)
    max_runtime = parse_duration(args.duration) if args.duration else 0.0
    trader = Trader(broker=broker, chromosome=chromosome, max_runtime_sec=max_runtime)
    if max_runtime:
        print(f"Paper trading against live MetaApi feed for {args.duration} (Ctrl+C to stop early).")
    else:
        print("Paper trading against live MetaApi feed. Ctrl+C to stop.")
    trader.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
