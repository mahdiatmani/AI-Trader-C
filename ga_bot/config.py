"""Central configuration for the GA trading bot.

All tunables live here so the same constants are used by training, paper
trading, and (later) live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "ga_bot" / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class InstrumentConfig:
    # XM Global lists spot gold as `GOLD`, not `XAUUSD`. The MetaApi
    # live broker will use this exact name when placing orders.
    symbol: str = "GOLD"
    # 1 standard lot = 100 oz of gold. We model positions in *units* (oz)
    # so risk math is independent of broker lot conventions.
    contract_size: float = 100.0
    # Quote currency per 1.00 price move per 1 unit (oz). For gold this
    # is exactly 1 USD per 1 USD move per ounce (verified via
    # symbol_info.trade_tick_value / trade_tick_size on XM).
    value_per_point: float = 1.0
    # Real XM spread on GOLD observed via symbol_info.spread (~48 points
    # at 0.01/point = $0.48). Slightly conservative.
    spread: float = 0.50
    # Commission per round-turn per ounce. XM Global standard accounts
    # are spread-only — keep at 0 unless you're on a Zero account.
    commission_per_unit: float = 0.0
    # Minimum stop distance the broker will accept (price units).
    min_stop_distance: float = 1.0


@dataclass
class AccountConfig:
    starting_balance: float = 250.0
    leverage: int = 10000  # User-requested 1:10000.
    # Hard cap so a single trade can never use more than this fraction of
    # equity as margin, even with extreme leverage.
    max_margin_fraction: float = 0.50
    # Daily drawdown circuit breaker (fraction of starting day equity).
    daily_dd_limit: float = 0.05


@dataclass
class GAConfig:
    population_size: int = 80
    max_generations: int = 200
    elite_count: int = 4
    tournament_size: int = 4
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15      # per-gene probability
    mutation_sigma: float = 0.15     # gaussian step size in normalized [0,1] space
    random_seed: int = 42

    # Stopping criteria — training halts when ALL of these are met on
    # the **TEST** split (fully held out, the GA never selects on it).
    # Win rate is intentionally NOT in this list. The user's goal is
    # live profit with capital preservation, not a pretty WR number.
    target_return_pct: float = 0.50          # +50 % on the held-out test slice
    max_acceptable_dd: float = 0.20          # never deeper than 20 % peak-to-trough
    max_worst_trade_pct: float = 0.05        # no single trade may lose >5 % of starting capital
    min_profit_factor_for_stop: float = 1.50 # gross win / gross loss ≥ 1.5
    min_trades_for_stop: int = 50


@dataclass
class BacktestConfig:
    # Risk per trade as a fraction of current equity. Used as a *cap* —
    # the chromosome's own gene can shrink it further.
    max_risk_per_trade: float = 0.02
    # Slippage in price units applied at fills (in addition to spread).
    slippage: float = 0.05
    # If both SL and TP could plausibly fill in the same bar, assume the
    # SL hit first (conservative).
    pessimistic_fills: bool = True
    # Walk-forward 60 / 20 / 20 split (oldest → newest):
    #   train_split  →  used to backtest each chromosome (the GA *sees*
    #                   this slice during evolution)
    #   val_split    →  also backtested for every chromosome, fed into
    #                   the fitness function as the "out-of-sample"
    #                   signal that penalizes overfit
    #   the remaining 1 - train - val is the **test set**: completely
    #   held out, used only for the stop criterion. The GA never sees
    #   it during selection.
    train_split: float = 0.60
    val_split: float = 0.20
    # The train slice is further chopped into N contiguous chronological
    # sub-folds. Fitness uses the WORST fold + val as the bottleneck for
    # every base metric (return, DD, PF, worst trade). This forces the
    # GA to find a chromosome that survives multiple sub-regimes inside
    # train, not one that overfits to a single trend within it.
    n_train_folds: int = 4


@dataclass
class TradingConfig:
    timeframe_minutes: int = 5
    # How long to wait between bar polls in the live/paper loop.
    poll_seconds: int = 15
    # Max concurrent open positions.
    max_open_positions: int = 1


@dataclass
class Config:
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)


CONFIG = Config()
