"""Load XAUUSD historical OHLCV data from CSV.

Supported formats:
    1. Standard CSV with a single timestamp column:
        timestamp,open,high,low,close,volume
        2024-01-02 00:00:00,2065.50,2066.10,2065.20,2065.80,123
    2. MetaTrader 5 export with separate Date / Time columns:
        Date,Time,Open,High,Low,Close,Volume
        2024.01.02,00:00,2065.50,2066.10,2065.20,2065.80,123

Drop your CSV in `./data/` and pass the file name to `load_csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import CONFIG, DATA_DIR


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "date" in df.columns and "time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="mixed",
            errors="coerce",
        )
        df = df.drop(columns=["date", "time"])
    if "timestamp" not in df.columns:
        # Try the first column as the timestamp.
        first = df.columns[0]
        df = df.rename(columns={first: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    if "volume" not in df.columns:
        df["volume"] = 0.0

    needed = ["open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"CSV is missing required column: {col}")
    return df[needed].astype(float)


def load_csv(filename: str) -> pd.DataFrame:
    path = Path(filename)
    if not path.is_absolute():
        path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Place your XAUUSD M5 CSV in {DATA_DIR} (e.g. XAUUSD_M5.csv)."
        )
    raw = pd.read_csv(path)
    return _normalize_columns(raw)


def split_train_val(
    df: pd.DataFrame, train_fraction: float = CONFIG.backtest.train_split
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Two-way walk-forward split (kept for backwards compat)."""
    n = len(df)
    cut = int(n * train_fraction)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def split_train_val_test(
    df: pd.DataFrame,
    train_fraction: float = CONFIG.backtest.train_split,
    val_fraction: float = CONFIG.backtest.val_split,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way walk-forward split (oldest → newest).

    Train and validation are both seen by the GA (train as the fitness
    backbone, val as the overfit check). Test is completely held out
    and only the stop criterion looks at it.
    """
    n = len(df)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    if val_end >= n:
        val_end = n - 1
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )
