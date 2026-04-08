# GA Gold Trader — Workflow

End-to-end pipeline for the Genetic-Algorithm XAUUSD bot, from a fresh
clone all the way to live execution. Each phase has a **gate** you must
pass before moving on.

```
┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐
│ 0. Setup │──▶│ 1. Data  │──▶│ 2. Train GA  │──▶│ 3. Paper     │──▶│ 4. Live  │
└──────────┘   └──────────┘   └──────────────┘   └──────────────┘   └──────────┘
   env+deps      CSV in        WR ≥ 80% on        ≥ 2 weeks on        gated by
                 ./data/       validation         live MetaApi        --i-understand
                               + PF ≥ 1.20        feed, profitable    -the-risk
                               + ≥ 50 trades
```

---

## Phase 0 — Setup

**Goal:** working Python env with the package importable.

```bash
cd "AI Trader"
python -m venv .venv
# Windows
.venv\Scripts\activate
# Ubuntu
source .venv/bin/activate

pip install -r requirements.txt
```

Sanity check:

```bash
python -c "from ga_bot.genetic_algorithm import GeneticAlgorithm; print('ok')"
```

**Gate:** import prints `ok` with no traceback.

---

## Phase 1 — Get historical data

**Goal:** at least 1–2 years of XAUUSD M5 OHLCV in `./data/`.

Two accepted CSV layouts (`ga_bot/data_loader.py` handles both):

1. **Standard**

   ```
   timestamp,open,high,low,close,volume
   2024-01-02 00:00:00,2065.50,2066.10,2065.20,2065.80,123
   ```

2. **MetaTrader 5 export**

   ```
   Date,Time,Open,High,Low,Close,Volume
   2024.01.02,00:00,2065.50,2066.10,2065.20,2065.80,123
   ```

How to obtain it:

- MT5 terminal → *Tools → History Center → XAUUSD M5 → Export*.
- Or pull it once via MetaApi and save to CSV.

Drop the file into `./data/XAUUSD_M5.csv`.

**Gate:**

```bash
python -c "from ga_bot.data_loader import load_csv; df=load_csv('XAUUSD_M5.csv'); print(len(df), df.index.min(), df.index.max())"
```

You should see at least ~50,000 bars and a date range that covers
several market regimes (bull, range, high-vol news periods).

---

## Phase 2 — Train the GA

**Goal:** evolve a chromosome whose **validation** win rate hits 80 %.

```bash
python train_ga.py --csv XAUUSD_M5.csv
```

Useful overrides:

```bash
python train_ga.py --csv XAUUSD_M5.csv \
    --population 120 \
    --generations 400 \
    --target-win-rate 0.80
```

What happens internally:

1. `data_loader.split_train_val` cuts the CSV into 80 % train / 20 % validation (walk-forward — validation is the *latest* slice).
2. `GeneticAlgorithm` initializes a random population of chromosomes.
3. Each generation:
   - every chromosome is backtested on the **train** split
   - `fitness.score` blends win rate, profit factor, drawdown, activity
   - elite carry over, the rest are made by tournament-4 selection + uniform crossover + Gaussian mutation
4. The best chromosome is then **re-tested on the held-out validation split**.
5. Training **stops automatically** the first generation where, on validation:
   - `win_rate ≥ 0.80`, **and**
   - `trades  ≥ 50`,   **and**
   - `profit_factor ≥ 1.20`
6. The winning chromosome is written to `ga_bot/models/best_chromosome.json`.

You'll see one line per generation:

```
gen 042 | fit=  3.412 | train wr= 81.3% pf= 1.74 n= 312 | val wr= 80.4% pf= 1.31 n=  74 |  6.8s
```

**Gate:** the final printout reads
`Target win rate reached. Ready for paper trading.`
If it says *NOT reached*, see *Troubleshooting* below.

---

## Phase 3 — Paper trading

**Goal:** confirm the saved model behaves on out-of-sample data and
**then** on a live feed, before risking a cent.

### 3a. Offline replay (do this first)

```bash
python run_paper.py --replay XAUUSD_M5.csv
```

This streams the CSV bar-by-bar through the `PaperBroker` exactly as
the live loop would. You'll see opens / closes / SL+TP hits in
`logs/paper_journal.jsonl`. At the end it prints final equity.

**Gate:** final equity ≥ starting equity, and the journal shows a
healthy distribution of wins/losses (not 1 trade per week, not 200 per
day).

### 3b. Live feed, paper execution (the real test)

On the Ubuntu cloud box:

```bash
pip install metaapi-cloud-sdk
export METAAPI_TOKEN=...
export METAAPI_ACCOUNT_ID=...

python run_paper.py --live-feed
```

This pulls real prices from MetaApi but routes every order through
`PaperBroker`, so the account balance never moves. Run it for at least
**two weeks** across different market conditions.

**Gate:**
- positive PnL on the journal,
- realized win rate ≥ 70 % (some slippage vs backtest is normal),
- max drawdown stayed under your daily-DD limit (5 % default),
- no unhandled exceptions in the loop.

---

## Phase 4 — Live trading

**Goal:** real money, small size, full safety rails.

```bash
python run_live.py --i-understand-the-risk
```

The `--i-understand-the-risk` flag is intentional — `run_live.py`
refuses to start without it. Before launching:

- account is funded with the **same** $250 you trained against (don't
  scale up the first day),
- leverage on the broker side matches `account.leverage` in
  `ga_bot/config.py` (1:10000),
- `metaapi-cloud-sdk` installed and credentials in env vars,
- you've set up some way to be alerted (e.g. tail
  `logs/paper_journal.jsonl` from a tmux session, or wire it to a
  Telegram bot later).

**Kill switches that already exist:**
- Daily drawdown circuit breaker (`account.daily_dd_limit = 0.05`) —
  stops opening new trades for the rest of the day.
- Margin guard (`account.max_margin_fraction = 0.50`) — refuses any
  order that would consume more than 50 % of equity as margin.
- Mandatory SL on every order — set from the chromosome's
  `sl_atr_mult`.
- One position at a time (`trading.max_open_positions = 1`).

---

## File reference

| File                              | Role                                                  |
|-----------------------------------|-------------------------------------------------------|
| `train_ga.py`                     | Phase 2 entry point                                    |
| `run_paper.py`                    | Phase 3 entry point (replay or live feed)              |
| `run_live.py`                     | Phase 4 entry point (gated)                            |
| `ga_bot/config.py`                | All knobs — leverage, risk, GA params, stop criteria   |
| `ga_bot/chromosome.py`            | 22-gene encoding, JSON save/load                       |
| `ga_bot/strategy.py`              | Chromosome → indicator signal → discrete decision      |
| `ga_bot/backtester.py`            | Event-driven backtest used inside the GA               |
| `ga_bot/fitness.py`               | Scalar fitness blend                                   |
| `ga_bot/genetic_algorithm.py`     | Population loop, selection, crossover, mutation, stop  |
| `ga_bot/data_loader.py`           | CSV loader + walk-forward split                        |
| `ga_bot/trader.py`                | Live/paper loop, broker-agnostic                       |
| `ga_bot/broker/base.py`           | Abstract `Broker` interface                            |
| `ga_bot/broker/paper.py`          | Simulated broker for phases 3a + 3b                    |
| `ga_bot/broker/metaapi_live.py`   | MetaApi cloud SDK shim for phase 4                     |
| `ga_bot/models/best_chromosome.json` | Trained model artifact                              |
| `logs/paper_journal.jsonl`        | Append-only log of every paper-broker event            |

---

## Troubleshooting

**Training never hits 80 % WR.**
80 % WR on M5 gold is an aggressive target. Options, in order of preference:

1. Give it more data (2+ years instead of 6 months).
2. Increase `--population` and `--generations`.
3. Lower `--target-win-rate` to 0.70 — still healthy, far more reachable.
4. Add more indicators / genes in `chromosome.py` and `strategy.py`.

**Validation WR is high but trade count is low (< 50).**
The stop gate intentionally refuses these — they're statistically
meaningless. Either lower the chromosome's `signal_threshold` range in
`GENE_SCHEMA` to encourage more activity, or train on a longer history.

**Backtest is profitable, paper is not.**
Spread/slippage mismatch. Bump `instrument.spread` and
`backtest.slippage` in `config.py` to match what your broker actually
charges, then re-train.

**Live errors out on `metaapi_cloud_sdk` import.**
You haven't installed it (`pip install metaapi-cloud-sdk`). The
training and paper-replay paths don't need it — only `--live-feed` and
`run_live.py` do.

---

## Continuous improvement loop

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ collect new │──▶│  re-train   │──▶│ paper test  │
│   M5 data   │   │   the GA    │   │  the model  │
└─────────────┘   └─────────────┘   └─────────────┘
       ▲                                    │
       └────────────────────────────────────┘
              monthly cadence
```

Re-run training monthly with the latest history appended to your CSV.
Keep the previous `best_chromosome.json` (rename it with a date suffix)
so you can roll back if the new one underperforms in paper.
