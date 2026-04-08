"""Export historical M5 bars from a running MetaTrader 5 terminal to CSV.

Why this exists:
    Your Ubuntu cloud instance can't run the MT5 terminal (Windows-only).
    This script runs on your Windows box where MT5 is installed and
    logged in to XM, pulls a few years of M5 bars in chunks, and writes
    a CSV that the GA bot can train on. You then `git push` the CSV and
    `git pull` it on the Ubuntu box.

Usage:
    # From the project root, with MT5 terminal open and logged in:
    python scripts/export_mt5_history.py --symbol GOLD --years 2 --out data/XAUUSD_M5.csv

Notes:
    * XM Global lists spot gold as `GOLD`, not `XAUUSD` — that's the
      default here. Change with --symbol if your broker differs.
    * MT5's `copy_rates_range` is limited by `terminal.maxbars`
      (typically 100k). We chunk the request in 30-day windows so the
      total history is unbounded.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.stderr.write(
        "MetaTrader5 package not installed. Run `pip install MetaTrader5`.\n"
    )
    raise


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def fetch_chunked(symbol: str, start: datetime, end: datetime, chunk_days: int = 30) -> pd.DataFrame:
    """Pull [start, end] M5 bars in chunk_days windows, concatenated."""
    frames = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=chunk_days), end)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, cursor, window_end)
        if rates is None:
            err = mt5.last_error()
            print(f"  ! copy_rates_range failed for {cursor.date()}..{window_end.date()}: {err}")
        else:
            df = pd.DataFrame(rates)
            print(f"  {cursor.date()} .. {window_end.date()}  ->  {len(df):,} bars")
            if len(df):
                frames.append(df)
        cursor = window_end
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MT5 M5 history to CSV.")
    parser.add_argument("--symbol", default="GOLD", help="MT5 symbol name (XM uses GOLD for spot gold)")
    parser.add_argument("--years", type=float, default=2.0, help="How many years of history to pull")
    parser.add_argument("--out", default=str(DATA_DIR / "XAUUSD_M5.csv"), help="Output CSV path")
    parser.add_argument("--chunk-days", type=int, default=30, help="Days per copy_rates_range call")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"MT5 initialize() failed: {mt5.last_error()}", file=sys.stderr)
        return 1

    try:
        if not mt5.symbol_select(args.symbol, True):
            print(f"Symbol {args.symbol!r} not available on this broker.", file=sys.stderr)
            return 2

        info = mt5.symbol_info(args.symbol)
        tick = mt5.symbol_info_tick(args.symbol)
        print(
            f"Broker symbol: {info.name}  digits={info.digits}  "
            f"point={info.point}  contract_size={info.trade_contract_size}  "
            f"current bid/ask={tick.bid}/{tick.ask}"
        )

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(args.years * 365))
        print(f"Fetching {args.symbol} M5 from {start.date()} to {end.date()} "
              f"in {args.chunk_days}-day chunks ...")

        df = fetch_chunked(args.symbol, start, end, args.chunk_days)
        if df.empty:
            print("No data returned. Make sure the MT5 terminal is open and logged in.", file=sys.stderr)
            return 3

        # Convert MT5 unix-seconds 'time' to a real timestamp column.
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        # data_loader.py accepts the standard layout: timestamp,open,high,low,close,volume
        out_df = pd.DataFrame({
            "timestamp": df["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S"),
            "open":   df["open"],
            "high":   df["high"],
            "low":    df["low"],
            "close":  df["close"],
            "volume": df["tick_volume"],
        })

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        print(
            f"\nWrote {len(out_df):,} rows to {out_path}\n"
            f"  first: {out_df['timestamp'].iloc[0]}\n"
            f"  last : {out_df['timestamp'].iloc[-1]}\n"
            f"  size : {out_path.stat().st_size / 1024 / 1024:.2f} MB"
        )
    finally:
        mt5.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
