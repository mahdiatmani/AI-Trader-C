"""Chromosome = the evolvable parameter set for a trading strategy.

Each gene is stored in normalized [0, 1] space so the GA's crossover and
mutation operators are uniform across genes. The `decoded` view maps each
gene back to its real-world range and type (int / float).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# (name, low, high, kind)  — kind in {"int", "float"}
GENE_SCHEMA: List[Tuple[str, float, float, str]] = [
    # Trend / momentum periods
    ("ema_fast",        5,    20,   "int"),
    ("ema_slow",        21,   100,  "int"),
    ("ema_trend",       100,  300,  "int"),
    ("rsi_period",      5,    30,   "int"),
    ("rsi_buy_th",      10.0, 40.0, "float"),
    ("rsi_sell_th",     60.0, 90.0, "float"),
    ("bb_period",       10,   40,   "int"),
    ("bb_std",          1.0,  3.0,  "float"),
    ("macd_fast",       5,    20,   "int"),
    ("macd_slow",       21,   60,   "int"),
    ("macd_signal",     5,    15,   "int"),
    ("atr_period",      5,    30,   "int"),

    # Risk / sizing
    ("sl_atr_mult",     0.5,  5.0,  "float"),
    ("tp_atr_mult",     1.0,  10.0, "float"),
    ("risk_pct",        0.25, 2.0,  "float"),  # % of equity per trade
    ("min_atr_pct",     0.0,  0.005,"float"),  # skip if ATR/price below this

    # Signal weights — combined to a single score in [-1, 1]
    ("w_ema_cross",     -1.0, 1.0,  "float"),
    ("w_rsi",           -1.0, 1.0,  "float"),
    ("w_bb",            -1.0, 1.0,  "float"),
    ("w_macd",          -1.0, 1.0,  "float"),
    ("w_trend",         -1.0, 1.0,  "float"),

    # Score threshold to actually take a trade
    ("signal_threshold",0.10, 0.80, "float"),
]

GENE_NAMES = [g[0] for g in GENE_SCHEMA]
N_GENES = len(GENE_SCHEMA)


@dataclass
class Chromosome:
    genes: np.ndarray  # shape (N_GENES,), values in [0, 1]
    fitness: float = -np.inf
    metrics: Dict[str, float] = field(default_factory=dict)

    # ---------- construction ----------
    @classmethod
    def random(cls, rng: np.random.Generator) -> "Chromosome":
        return cls(genes=rng.random(N_GENES))

    @classmethod
    def from_decoded(cls, decoded: Dict[str, float]) -> "Chromosome":
        genes = np.zeros(N_GENES)
        for i, (name, lo, hi, _kind) in enumerate(GENE_SCHEMA):
            v = float(decoded[name])
            genes[i] = (v - lo) / (hi - lo)
        return cls(genes=np.clip(genes, 0.0, 1.0))

    # ---------- decoding ----------
    def decoded(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for i, (name, lo, hi, kind) in enumerate(GENE_SCHEMA):
            raw = lo + float(self.genes[i]) * (hi - lo)
            out[name] = int(round(raw)) if kind == "int" else float(raw)
        # Enforce structural constraints between related genes.
        if out["ema_slow"] <= out["ema_fast"]:
            out["ema_slow"] = out["ema_fast"] + 1
        if out["macd_slow"] <= out["macd_fast"]:
            out["macd_slow"] = out["macd_fast"] + 1
        if out["rsi_sell_th"] <= out["rsi_buy_th"] + 5:
            out["rsi_sell_th"] = out["rsi_buy_th"] + 5
        return out

    # ---------- persistence ----------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "genes": self.genes.tolist(),
            "fitness": float(self.fitness),
            "metrics": self.metrics,
            "decoded": self.decoded(),
            "schema": [list(g) for g in GENE_SCHEMA],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chromosome":
        c = cls(genes=np.asarray(d["genes"], dtype=float))
        c.fitness = float(d.get("fitness", -np.inf))
        c.metrics = dict(d.get("metrics", {}))
        return c

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Chromosome":
        return cls.from_dict(json.loads(Path(path).read_text()))
