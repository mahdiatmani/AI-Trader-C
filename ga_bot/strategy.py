"""Apply a chromosome to OHLCV data and produce trading signals.

A signal is a single number in [-1, 1]:
    +1   strong long
     0   neutral
    -1   strong short
The chromosome's `signal_threshold` then turns it into a discrete decision.

The same `Strategy` is used by:
    * the backtester (vectorized over a full DataFrame)
    * the live trader (called on each new closed bar)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from . import indicators as ind
from .chromosome import Chromosome


@dataclass
class StrategyOutput:
    signal: pd.Series           # combined score in [-1, 1]
    atr: pd.Series              # ATR per bar (for SL/TP sizing)
    decision: pd.Series         # +1 long, -1 short, 0 flat
    skip_low_vol: pd.Series     # bool, True where ATR/price below threshold


class Strategy:
    def __init__(self, chromosome: Chromosome):
        self.chromosome = chromosome
        self.params: Dict = chromosome.decoded()

    # ----- vectorized path used by the backtester -----
    def compute(self, df: pd.DataFrame) -> StrategyOutput:
        p = self.params
        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema_fast = ind.ema(close, p["ema_fast"])
        ema_slow = ind.ema(close, p["ema_slow"])
        ema_trend = ind.ema(close, p["ema_trend"])
        rsi_v = ind.rsi(close, p["rsi_period"])
        bb_lo, bb_mid, bb_up = ind.bollinger(close, p["bb_period"], p["bb_std"])
        _, _, macd_hist = ind.macd(
            close, p["macd_fast"], p["macd_slow"], p["macd_signal"]
        )
        atr_v = ind.atr(high, low, close, p["atr_period"])

        # ---- normalize each sub-signal into [-1, 1] ----
        s_ema = np.tanh((ema_fast - ema_slow) / atr_v.replace(0.0, np.nan)).fillna(0.0)

        # RSI: -1 at overbought, +1 at oversold
        rsi_centered = (50.0 - rsi_v) / 50.0
        s_rsi = rsi_centered.clip(-1.0, 1.0)

        # Bollinger position: -1 above upper band, +1 below lower band
        bb_range = (bb_up - bb_lo).replace(0.0, np.nan)
        s_bb = ((bb_mid - close) / (bb_range / 2.0)).clip(-1.0, 1.0).fillna(0.0)

        s_macd = np.tanh(macd_hist / atr_v.replace(0.0, np.nan)).fillna(0.0)

        s_trend = np.tanh((close - ema_trend) / atr_v.replace(0.0, np.nan)).fillna(0.0)

        # ---- combined weighted score ----
        weights = np.array(
            [p["w_ema_cross"], p["w_rsi"], p["w_bb"], p["w_macd"], p["w_trend"]],
            dtype=float,
        )
        wsum = float(np.sum(np.abs(weights))) or 1.0
        weights = weights / wsum

        score = (
            weights[0] * s_ema
            + weights[1] * s_rsi
            + weights[2] * s_bb
            + weights[3] * s_macd
            + weights[4] * s_trend
        ).clip(-1.0, 1.0)

        threshold = p["signal_threshold"]
        decision = pd.Series(0, index=df.index, dtype=int)
        decision[score >= threshold] = 1
        decision[score <= -threshold] = -1

        skip = (atr_v / close.replace(0.0, np.nan)) < p["min_atr_pct"]
        skip = skip.fillna(True)
        decision[skip] = 0

        return StrategyOutput(
            signal=score, atr=atr_v, decision=decision, skip_low_vol=skip
        )
