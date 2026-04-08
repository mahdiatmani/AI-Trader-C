"""Paper broker.

Holds a virtual balance and simulates fills against a price feed. The
price feed itself is supplied externally — usually a CSV replay during
development, or a thin live-data adapter (MetaApi quote stream) once
you're ready to trade against real prices without risking capital.

This way the *exact* same broker simulation runs whether you're testing
offline or shadowing a live feed.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from ..config import CONFIG, LOGS_DIR
from ..journal import JsonlJournal
from .base import Broker, Order, Position, Side

# A bar feed is any zero-arg callable that returns the most recent bars
# as a DataFrame indexed by timestamp with open/high/low/close/volume.
BarFeed = Callable[[str, int, int], pd.DataFrame]


class PaperBroker(Broker):
    def __init__(
        self,
        bar_feed: BarFeed,
        starting_balance: Optional[float] = None,
        journal_path: Optional[Path] = None,
    ):
        self._bar_feed = bar_feed
        self._balance = starting_balance or CONFIG.account.starting_balance
        self._equity = self._balance
        self._positions: List[Position] = []
        self._journal_path = Path(journal_path) if journal_path else LOGS_DIR / "paper_journal.jsonl"
        self._journal = JsonlJournal(self._journal_path)

    # ----- lifecycle -----
    def connect(self) -> None:
        self._log({"event": "connect", "ts": datetime.utcnow().isoformat(), "balance": self._balance})

    def disconnect(self) -> None:
        self._log({"event": "disconnect", "ts": datetime.utcnow().isoformat(), "balance": self._balance})

    # ----- account -----
    def get_balance(self) -> float:
        return self._balance

    def get_equity(self) -> float:
        # Mark-to-market open positions
        eq = self._balance
        if self._positions:
            last = self._latest_close()
            for p in self._positions:
                eq += int(p.side) * (last - p.entry_price) * p.units * CONFIG.instrument.value_per_point
        self._equity = eq
        return eq

    def get_open_positions(self) -> List[Position]:
        return list(self._positions)

    # ----- data -----
    def get_latest_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        return self._bar_feed(symbol, timeframe_minutes, count)

    def _latest_close(self) -> float:
        bars = self._bar_feed(CONFIG.instrument.symbol, CONFIG.trading.timeframe_minutes, 1)
        return float(bars["close"].iloc[-1])

    # ----- orders -----
    def place_order(self, order: Order) -> Optional[Position]:
        if len(self._positions) >= CONFIG.trading.max_open_positions:
            return None

        last = self._latest_close()
        spread = CONFIG.instrument.spread
        slip = CONFIG.backtest.slippage
        side = int(order.side)
        entry_price = last + side * (spread / 2.0 + slip)

        notional = order.units * entry_price
        margin = notional / max(CONFIG.account.leverage, 1)
        if margin > self._balance * CONFIG.account.max_margin_fraction:
            self._log({
                "event": "rejected", "reason": "margin",
                "required": margin, "balance": self._balance,
            })
            return None

        pos = Position(
            id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=Side(side),
            units=order.units,
            entry_price=entry_price,
            sl=order.sl,
            tp=order.tp,
            opened_at=datetime.utcnow(),
        )
        self._positions.append(pos)
        self._log({
            "event": "open", "id": pos.id, "side": int(pos.side),
            "units": pos.units, "entry": pos.entry_price,
            "sl": pos.sl, "tp": pos.tp,
            "ts": pos.opened_at.isoformat(),
        })
        return pos

    def close_position(self, position_id: str) -> None:
        for i, p in enumerate(self._positions):
            if p.id == position_id:
                last = self._latest_close()
                fill = last - int(p.side) * CONFIG.backtest.slippage
                pnl = int(p.side) * (fill - p.entry_price) * p.units * CONFIG.instrument.value_per_point
                self._balance += pnl
                self._log({
                    "event": "close", "id": p.id, "exit": fill,
                    "pnl": pnl, "balance": self._balance,
                    "ts": datetime.utcnow().isoformat(),
                })
                del self._positions[i]
                return

    # ----- SL/TP sweep, called by the trader on every tick -----
    def check_sl_tp(self) -> None:
        if not self._positions:
            return
        bars = self._bar_feed(CONFIG.instrument.symbol, CONFIG.trading.timeframe_minutes, 1)
        if bars.empty:
            return
        bar = bars.iloc[-1]
        high, low = float(bar["high"]), float(bar["low"])
        for p in list(self._positions):
            hit_sl = (p.side == Side.BUY and low <= p.sl) or (p.side == Side.SELL and high >= p.sl)
            hit_tp = (p.side == Side.BUY and high >= p.tp) or (p.side == Side.SELL and low <= p.tp)
            if hit_sl or hit_tp:
                # Pessimistic: if both, prefer SL.
                exit_price = p.sl if hit_sl else p.tp
                pnl = int(p.side) * (exit_price - p.entry_price) * p.units * CONFIG.instrument.value_per_point
                self._balance += pnl
                self._log({
                    "event": "sltp_close", "id": p.id, "reason": "sl" if hit_sl else "tp",
                    "exit": exit_price, "pnl": pnl, "balance": self._balance,
                    "ts": datetime.utcnow().isoformat(),
                })
                self._positions.remove(p)

    # ----- internal -----
    def _log(self, payload: dict) -> None:
        self._journal.write(payload)
