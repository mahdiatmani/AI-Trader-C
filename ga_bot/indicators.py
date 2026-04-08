"""Vectorized technical indicators.

Kept dependency-light (numpy + pandas only) so the same code runs inside the
backtester loop and on a live bar feed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=max(1, int(period)), adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(max(1, int(period)), min_periods=1).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    period = max(2, int(period))
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    period = max(1, int(period))
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().bfill()


def bollinger(series: pd.Series, period: int, num_std: float):
    period = max(2, int(period))
    mid = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std(ddof=0).fillna(0.0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper


def macd(series: pd.Series, fast: int, slow: int, signal: int):
    fast = max(1, int(fast))
    slow = max(fast + 1, int(slow))
    signal = max(1, int(signal))
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def normalize(value: pd.Series) -> pd.Series:
    """Soft-normalize a series into roughly [-1, 1] using a rolling z-score."""
    win = 200
    mu = value.rolling(win, min_periods=20).mean()
    sd = value.rolling(win, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    z = (value - mu) / sd
    return z.clip(-3.0, 3.0).fillna(0.0) / 3.0
