"""Event-driven backtester used to evaluate a chromosome's fitness.

Models a single open position at a time, with hard SL/TP, conservative
intra-bar fills, spread, slippage and leverage-aware sizing. Output is a
dictionary of metrics that the fitness function consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import CONFIG
from .strategy import Strategy


@dataclass
class Trade:
    side: int            # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    units: float
    pnl: float
    reason: str          # "tp", "sl", "signal", "eod"


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    final_equity: float = 0.0

    def metrics(self) -> Dict[str, float]:
        n = len(self.trades)
        if n == 0:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "net_pnl": 0.0,
                "max_dd": 1.0,
                "expectancy": 0.0,
                "sharpe": 0.0,
            }
        pnls = np.array([t.pnl for t in self.trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = len(wins) / n
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

        eq = self.equity_curve.values if len(self.equity_curve) else np.array([CONFIG.account.starting_balance])
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(peak, 1e-9)
        max_dd = float(dd.max()) if len(dd) else 0.0

        rets = pd.Series(eq).pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252 * 24 * 12)) if rets.std() > 0 else 0.0

        return {
            "trades": n,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "net_pnl": float(pnls.sum()),
            "max_dd": max_dd,
            "expectancy": float(pnls.mean()),
            "sharpe": sharpe,
        }


class Backtester:
    def __init__(self, df: pd.DataFrame):
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Backtester needs columns {required}, missing {missing}")
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        self.cfg = CONFIG

    # ----- core simulation -----
    def run(self, strategy: Strategy) -> BacktestResult:
        df = self.df
        out = strategy.compute(df)
        decision = out.decision.values
        atr_arr = out.atr.values

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        index = df.index

        cfg_i = self.cfg.instrument
        cfg_a = self.cfg.account
        cfg_b = self.cfg.backtest

        spread = cfg_i.spread
        slip = cfg_b.slippage
        leverage = cfg_a.leverage
        max_marg_frac = cfg_a.max_margin_fraction

        risk_pct_cap = cfg_b.max_risk_per_trade
        chrom_risk = strategy.params["risk_pct"] / 100.0
        risk_pct = min(chrom_risk, risk_pct_cap)

        sl_mult = strategy.params["sl_atr_mult"]
        tp_mult = strategy.params["tp_atr_mult"]

        equity = cfg_a.starting_balance
        day_start_equity = equity
        current_day = None

        position = None  # dict with side, entry, sl, tp, units, entry_time
        trades: List[Trade] = []
        equity_curve = np.empty(len(df))

        for i in range(len(df)):
            ts = index[i]
            day = ts.date() if hasattr(ts, "date") else None
            if day != current_day:
                current_day = day
                day_start_equity = equity

            # Daily DD circuit breaker
            day_dd = (day_start_equity - equity) / max(day_start_equity, 1e-9)
            trading_blocked = day_dd >= cfg_a.daily_dd_limit

            high_i = highs[i]
            low_i = lows[i]
            close_i = closes[i]

            # ---- check exits on existing position ----
            if position is not None:
                hit_sl = False
                hit_tp = False
                if position["side"] == 1:
                    hit_sl = low_i <= position["sl"]
                    hit_tp = high_i >= position["tp"]
                else:
                    hit_sl = high_i >= position["sl"]
                    hit_tp = low_i <= position["tp"]

                exit_price = None
                reason = None
                if hit_sl and hit_tp:
                    if cfg_b.pessimistic_fills:
                        exit_price = position["sl"]; reason = "sl"
                    else:
                        exit_price = position["tp"]; reason = "tp"
                elif hit_sl:
                    exit_price = position["sl"]; reason = "sl"
                elif hit_tp:
                    exit_price = position["tp"]; reason = "tp"

                if exit_price is not None:
                    fill = exit_price - position["side"] * slip
                    pnl = position["side"] * (fill - position["entry"]) * position["units"] * cfg_i.value_per_point
                    pnl -= cfg_i.commission_per_unit * position["units"]
                    equity += pnl
                    trades.append(
                        Trade(
                            side=position["side"],
                            entry_time=position["entry_time"],
                            entry_price=position["entry"],
                            exit_time=ts,
                            exit_price=fill,
                            units=position["units"],
                            pnl=pnl,
                            reason=reason,
                        )
                    )
                    position = None

            # ---- consider new entry ----
            if position is None and not trading_blocked and i > 0:
                d = decision[i]
                a = atr_arr[i]
                if d != 0 and a > 0 and not np.isnan(a):
                    sl_dist = max(sl_mult * a, cfg_i.min_stop_distance)
                    tp_dist = max(tp_mult * a, cfg_i.min_stop_distance)

                    risk_dollars = equity * risk_pct
                    units = risk_dollars / (sl_dist * cfg_i.value_per_point)
                    units = max(units, 0.0)

                    # Leverage / margin guard
                    notional = units * close_i
                    required_margin = notional / max(leverage, 1)
                    if required_margin > equity * max_marg_frac:
                        units = (equity * max_marg_frac * leverage) / max(close_i, 1e-9)

                    if units > 0:
                        entry = close_i + d * (spread / 2.0 + slip)
                        sl = entry - d * sl_dist
                        tp = entry + d * tp_dist
                        position = {
                            "side": d,
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "units": units,
                            "entry_time": ts,
                        }

            equity_curve[i] = equity

        # Force-close any open position at the last bar
        if position is not None:
            last_close = closes[-1]
            fill = last_close - position["side"] * slip
            pnl = position["side"] * (fill - position["entry"]) * position["units"] * cfg_i.value_per_point
            equity += pnl
            trades.append(
                Trade(
                    side=position["side"],
                    entry_time=position["entry_time"],
                    entry_price=position["entry"],
                    exit_time=index[-1],
                    exit_price=fill,
                    units=position["units"],
                    pnl=pnl,
                    reason="eod",
                )
            )
            equity_curve[-1] = equity

        result = BacktestResult(
            trades=trades,
            equity_curve=pd.Series(equity_curve, index=index),
            final_equity=equity,
        )
        return result
