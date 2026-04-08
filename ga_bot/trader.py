"""Live / paper trading loop driven by a saved chromosome.

Reads the latest bars from the broker, runs the chromosome's strategy on
them, and places exactly one trade per bar transition. The same loop
works against `PaperBroker` and `MetaApiLiveBroker` because both
implement the `Broker` interface.

Long-running deployment helpers:
    * heartbeat — writes a JSON timestamp to logs/heartbeat.json on
      every tick so external watchdogs can verify the bot is alive
    * max_runtime — optional duration cap so the loop stops cleanly
      after N seconds (use parse_duration() for "7d" / "12h" / "30m")
    * graceful shutdown — handles SIGINT/SIGTERM so systemd `Restart=`
      and Ctrl+C both close the broker connection cleanly
"""

from __future__ import annotations

import json
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .broker.base import Broker, Order, Side
from .broker.paper import PaperBroker
from .chromosome import Chromosome
from .config import CONFIG, LOGS_DIR
from .strategy import Strategy


_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$", re.IGNORECASE)
_DURATION_FACTORS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str) -> float:
    """Parse '30s' / '15m' / '12h' / '7d' / '2w' into seconds."""
    if text is None:
        return 0.0
    m = _DURATION_RE.match(text)
    if not m:
        raise ValueError(f"invalid duration: {text!r} (use e.g. 30s, 15m, 12h, 7d, 2w)")
    return int(m.group(1)) * _DURATION_FACTORS[m.group(2).lower()]


class Trader:
    def __init__(
        self,
        broker: Broker,
        chromosome: Chromosome,
        dry_run: bool = False,
        max_runtime_sec: float = 0.0,
        heartbeat_path: Optional[Path] = None,
    ):
        self.broker = broker
        self.chromosome = chromosome
        self.strategy = Strategy(chromosome)
        self.dry_run = dry_run
        self.max_runtime_sec = float(max_runtime_sec)
        self.heartbeat_path = Path(heartbeat_path) if heartbeat_path else LOGS_DIR / "heartbeat.json"
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        self._last_bar_ts: Optional[pd.Timestamp] = None
        self._day_start_equity: Optional[float] = None
        self._current_day = None
        self._stop_requested = False
        self._tick_count = 0

    # ----- daily DD -----
    def _check_daily_dd(self) -> bool:
        equity = self.broker.get_equity()
        today = datetime.utcnow().date()
        if self._current_day != today:
            self._current_day = today
            self._day_start_equity = equity
        if self._day_start_equity is None:
            self._day_start_equity = equity
        dd = (self._day_start_equity - equity) / max(self._day_start_equity, 1e-9)
        return dd < CONFIG.account.daily_dd_limit

    # ----- order construction -----
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

    # ----- heartbeat -----
    def _write_heartbeat(self, status: str = "alive", note: str = "") -> None:
        payload = {
            "status": status,
            "ts": datetime.now(timezone.utc).isoformat(),
            "tick": self._tick_count,
            "balance": float(self.broker.get_balance()),
            "equity": float(self.broker.get_equity()),
            "open_positions": len(self.broker.get_open_positions()),
            "last_bar_ts": str(self._last_bar_ts) if self._last_bar_ts is not None else None,
            "note": note,
        }
        try:
            tmp = self.heartbeat_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.heartbeat_path)
        except OSError as exc:
            print(f"[trader] heartbeat write failed: {exc}")

    # ----- one tick -----
    def step(self) -> None:
        if isinstance(self.broker, PaperBroker):
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

        if self.broker.get_open_positions():
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

    # ----- main loop -----
    def request_stop(self, *_args) -> None:
        self._stop_requested = True
        print("[trader] stop requested")

    def run_forever(self) -> None:
        # Install signal handlers for clean systemd / Ctrl+C shutdown.
        try:
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGTERM, self.request_stop)
        except (ValueError, OSError):
            pass  # not in main thread (e.g. during tests)

        self.broker.connect()
        start = time.monotonic()
        self._write_heartbeat("starting")
        try:
            while not self._stop_requested:
                try:
                    self.step()
                except Exception as exc:  # noqa: BLE001 — never crash the loop
                    print(f"[trader] step error: {exc}")
                    self._write_heartbeat("error", note=str(exc))

                self._tick_count += 1
                self._write_heartbeat("alive")

                if self.max_runtime_sec > 0 and (time.monotonic() - start) >= self.max_runtime_sec:
                    print(f"[trader] reached max runtime ({self.max_runtime_sec:.0f}s) — stopping")
                    break

                # Sleep in small slices so SIGTERM is responsive.
                slept = 0.0
                while slept < CONFIG.trading.poll_seconds and not self._stop_requested:
                    time.sleep(min(0.5, CONFIG.trading.poll_seconds - slept))
                    slept += 0.5
        finally:
            self._write_heartbeat("stopped")
            self.broker.disconnect()
