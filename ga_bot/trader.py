"""Live / paper trading loop driven by a saved chromosome.

Reads the latest bars from the broker, runs the chromosome's strategy on
them, and places exactly one trade per bar transition. The same loop
works against `PaperBroker` and `MetaApiLiveBroker` because both
implement the `Broker` interface.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .broker.base import Broker, Order, Side
from .broker.paper import PaperBroker
from .chromosome import Chromosome
from .config import CONFIG
from .strategy import Strategy


class Trader:
    def __init__(self, broker: Broker, chromosome: Chromosome, dry_run: bool = False):
        self.broker = broker
        self.chromosome = chromosome
        self.strategy = Strategy(chromosome)
        self.dry_run = dry_run
        self._last_bar_ts: Optional[pd.Timestamp] = None
        self._day_start_equity: Optional[float] = None
        self._current_day = None

    def _check_daily_dd(self) -> bool:
        """Return True if we are still allowed to trade today."""
        equity = self.broker.get_equity()
        today = datetime.utcnow().date()
        if self._current_day != today:
            self._current_day = today
            self._day_start_equity = equity
        if self._day_start_equity is None:
            self._day_start_equity = equity
        dd = (self._day_start_equity - equity) / max(self._day_start_equity, 1e-9)
        return dd < CONFIG.account.daily_dd_limit

    def _build_order(self, last_close: float, atr_value: float, side: int) -> Optional[Order]:
        p = self.strategy.params
        sl_dist = max(p["sl_atr_mult"] * atr_value, CONFIG.instrument.min_stop_distance)
        tp_dist = max(p["tp_atr_mult"] * atr_value, CONFIG.instrument.min_stop_distance)

        equity = self.broker.get_equity()
        risk_pct = min(p["risk_pct"] / 100.0, CONFIG.backtest.max_risk_per_trade)
        risk_dollars = equity * risk_pct
        units = risk_dollars / max(sl_dist * CONFIG.instrument.value_per_point, 1e-9)

        notional = units * last_close
        margin = notional / max(CONFIG.account.leverage, 1)
        if margin > equity * CONFIG.account.max_margin_fraction:
            units = (equity * CONFIG.account.max_margin_fraction * CONFIG.account.leverage) / max(last_close, 1e-9)
        if units <= 0:
            return None

        sl = last_close - side * sl_dist
        tp = last_close + side * tp_dist
        return Order(
            symbol=CONFIG.instrument.symbol,
            side=Side.BUY if side > 0 else Side.SELL,
            units=units,
            sl=sl,
            tp=tp,
            comment="ga_bot",
        )

    def step(self) -> None:
        if isinstance(self.broker, PaperBroker):
            # Allow the paper broker to evaluate SL/TP against the latest bar.
            self.broker.check_sl_tp()

        bars = self.broker.get_latest_bars(
            CONFIG.instrument.symbol,
            CONFIG.trading.timeframe_minutes,
            count=max(400, self.strategy.params["ema_trend"] + 50),
        )
        if bars is None or len(bars) < 50:
            return

        last_ts = bars.index[-1]
        if self._last_bar_ts is not None and last_ts == self._last_bar_ts:
            return  # bar hasn't closed yet
        self._last_bar_ts = last_ts

        if not self._check_daily_dd():
            return

        out = self.strategy.compute(bars)
        decision = int(out.decision.iloc[-1])
        atr_value = float(out.atr.iloc[-1])
        last_close = float(bars["close"].iloc[-1])

        open_positions = self.broker.get_open_positions()
        if open_positions:
            return  # one position at a time

        if decision == 0 or atr_value <= 0 or np.isnan(atr_value):
            return

        order = self._build_order(last_close, atr_value, decision)
        if order is None:
            return

        if self.dry_run:
            print(f"[DRY] would place {order}")
            return

        self.broker.place_order(order)

    def run_forever(self) -> None:
        self.broker.connect()
        try:
            while True:
                try:
                    self.step()
                except Exception as exc:  # noqa: BLE001 — never crash the loop
                    print(f"[trader] step error: {exc}")
                time.sleep(CONFIG.trading.poll_seconds)
        finally:
            self.broker.disconnect()
