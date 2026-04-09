"""The GA engine: population init → evaluate → select → crossover → mutate.

Three-way walk-forward split, with the train slice further chopped
into N contiguous sub-folds for regime robustness:

    train (60%)  →  N=4 sub-folds, each backtested separately. Fitness
                    uses the WORST fold (worst return, deepest DD, etc)
                    so a chromosome that nails one regime in train but
                    fails on another can never win.
    val   (20%)  →  also backtested for every chromosome, treated as
                    the (N+1)-th slice in the worst-of-all aggregation.
    test  (20%)  →  fully held out. Only the stop criterion looks at it.

The stop check fires the moment the **test** slice produces enough
trades AND a target return % AND a max drawdown under the cap AND a
worst single-trade loss under the cap AND a profit factor above the
floor. Win rate is intentionally not part of this — the user's goal is
live profit with capital preservation, not a pretty WR number.

Stagnation injection: if best fitness fails to improve by > 1e-3 for
``stagnation_patience`` consecutive generations, 30 % of the population
gets replaced with fresh random chromosomes (elites are preserved).
This is the safety net for the kind of fitness lock the user hit on
the Ubuntu box at gen 89→153.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtester import Backtester
from .chromosome import N_GENES, Chromosome
from .config import CONFIG, MODELS_DIR
from .fitness import evaluate
from .strategy import Strategy

STAGNATION_PATIENCE = 15
STAGNATION_INJECT_FRACTION = 0.40
STAGNATION_IMPROVEMENT_EPS = 1e-3


@dataclass
class GenerationLog:
    generation: int
    best_fitness: float
    best_train_metrics: Dict[str, float]
    best_val_metrics: Dict[str, float]
    best_test_metrics: Dict[str, float]
    elapsed_sec: float


class GeneticAlgorithm:
    def __init__(
        self,
        train_folds: List[pd.DataFrame],
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        on_generation: Optional[Callable[[GenerationLog], None]] = None,
    ):
        if not train_folds:
            raise ValueError("train_folds must contain at least one DataFrame")
        self.cfg = CONFIG.ga
        self.rng = np.random.default_rng(self.cfg.random_seed)
        self.train_bts: List[Backtester] = [Backtester(df) for df in train_folds]
        self.val_bt = Backtester(val_df)
        self.test_bt = Backtester(test_df)
        self.on_generation = on_generation
        self.history: List[GenerationLog] = []

    # ---------- evolution operators ----------
    def _init_population(self) -> List[Chromosome]:
        return [Chromosome.random(self.rng) for _ in range(self.cfg.population_size)]

    def _tournament(self, pop: List[Chromosome]) -> Chromosome:
        idxs = self.rng.integers(0, len(pop), size=self.cfg.tournament_size)
        contenders = [pop[i] for i in idxs]
        return max(contenders, key=lambda c: c.fitness)

    def _crossover(self, a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if self.rng.random() > self.cfg.crossover_rate:
            return Chromosome(genes=a.genes.copy()), Chromosome(genes=b.genes.copy())
        # Uniform crossover with a per-gene blend
        mask = self.rng.random(N_GENES) < 0.5
        alpha = self.rng.random(N_GENES)
        child1 = np.where(mask, a.genes, alpha * a.genes + (1 - alpha) * b.genes)
        child2 = np.where(mask, b.genes, alpha * b.genes + (1 - alpha) * a.genes)
        return (
            Chromosome(genes=np.clip(child1, 0.0, 1.0)),
            Chromosome(genes=np.clip(child2, 0.0, 1.0)),
        )

    def _mutate(self, c: Chromosome) -> Chromosome:
        mask = self.rng.random(N_GENES) < self.cfg.mutation_rate
        noise = self.rng.normal(0.0, self.cfg.mutation_sigma, size=N_GENES)
        new_genes = np.where(mask, c.genes + noise, c.genes)
        return Chromosome(genes=np.clip(new_genes, 0.0, 1.0))

    # ---------- evaluation ----------
    def _evaluate(self, c: Chromosome) -> None:
        strat = Strategy(c)
        train_results = [bt.run(strat) for bt in self.train_bts]
        val_result = self.val_bt.run(strat)
        m = evaluate(train_results, val_result)
        c.fitness = m["fitness"]
        c.metrics = m

    def _test(self, c: Chromosome) -> Dict[str, float]:
        result = self.test_bt.run(Strategy(c))
        return result.metrics()

    # ---------- stopping rule ----------
    def _meets_stop_criteria(self, test_metrics: Dict[str, float]) -> bool:
        return (
            test_metrics["trades"] >= self.cfg.min_trades_for_stop
            and test_metrics["return_pct"] >= self.cfg.target_return_pct
            and test_metrics["max_dd"] <= self.cfg.max_acceptable_dd
            and test_metrics["worst_trade_pct"] <= self.cfg.max_worst_trade_pct
            and test_metrics["profit_factor"] >= self.cfg.min_profit_factor_for_stop
        )

    # ---------- main loop ----------
    def run(self, save_path: Optional[Path] = None) -> Chromosome:
        save_path = Path(save_path) if save_path else MODELS_DIR / "best_chromosome.json"
        pop = self._init_population()
        for c in pop:
            self._evaluate(c)
        pop.sort(key=lambda c: c.fitness, reverse=True)

        best_overall: Chromosome = pop[0]
        last_improvement_gen = 0

        for gen in range(1, self.cfg.max_generations + 1):
            t0 = time.time()

            # ----- breed new population -----
            new_pop: List[Chromosome] = pop[: self.cfg.elite_count]  # elitism
            while len(new_pop) < self.cfg.population_size:
                p1 = self._tournament(pop)
                p2 = self._tournament(pop)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.cfg.population_size:
                    new_pop.append(c2)

            # ----- stagnation injection: guided perturbation + random -----
            # Pure random injection failed (gen-300 run had 36-gen
            # plateaus) because random chromosomes start too far behind
            # the current elites. Fix: 70% of injected slots get a
            # *heavily perturbed* copy of a top-10 chromosome (explores
            # the neighborhood of known-good solutions), 30% stay random
            # (maintains diversity for escaping local basins entirely).
            stagnated_for = gen - last_improvement_gen
            if stagnated_for >= STAGNATION_PATIENCE:
                inject_n = int(self.cfg.population_size * STAGNATION_INJECT_FRACTION)
                if inject_n > 0:
                    n_guided = int(inject_n * 0.7)
                    n_random = inject_n - n_guided
                    top_k = min(10, len(pop))
                    inject_sigma = self.cfg.mutation_sigma * 3.0
                    idx = self.cfg.population_size - inject_n
                    for j in range(n_guided):
                        parent = pop[self.rng.integers(0, top_k)]
                        noise = self.rng.normal(0.0, inject_sigma, size=N_GENES)
                        new_pop[idx + j] = Chromosome(
                            genes=np.clip(parent.genes + noise, 0.0, 1.0)
                        )
                    for j in range(n_random):
                        new_pop[idx + n_guided + j] = Chromosome.random(self.rng)
                last_improvement_gen = gen  # reset patience clock

            # ----- evaluate (skip already-evaluated elites) -----
            for c in new_pop[self.cfg.elite_count :]:
                self._evaluate(c)
            new_pop.sort(key=lambda c: c.fitness, reverse=True)
            pop = new_pop

            best = pop[0]
            test_metrics = self._test(best)

            if best.fitness > best_overall.fitness + STAGNATION_IMPROVEMENT_EPS:
                best_overall = best
                last_improvement_gen = gen

            # The GenerationLog still wants the same fields the dashboard
            # / printer expect. The bare ``train_*`` keys in best.metrics
            # are already the WORST fold (set by fitness.evaluate), so
            # the train view shows the bottleneck the GA optimized
            # against. val_* keys are val.
            train_view = {
                k.replace("train_", ""): v
                for k, v in best.metrics.items()
                if k.startswith("train_") and not k.startswith("train_fold")
            }
            # If for some reason no bare train_* keys exist, fall back
            # to fold 0.
            if not train_view:
                train_view = {
                    k.replace("train_fold0_", ""): v
                    for k, v in best.metrics.items()
                    if k.startswith("train_fold0_")
                }
            val_view = {
                k.replace("val_", ""): v
                for k, v in best.metrics.items()
                if k.startswith("val_")
            }

            log = GenerationLog(
                generation=gen,
                best_fitness=best.fitness,
                best_train_metrics=train_view,
                best_val_metrics=val_view,
                best_test_metrics=test_metrics,
                elapsed_sec=time.time() - t0,
            )
            self.history.append(log)
            if self.on_generation is not None:
                self.on_generation(log)

            # ----- stop check (TEST only) -----
            if self._meets_stop_criteria(test_metrics):
                best.metrics = {
                    **best.metrics,
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
                best.save(save_path)
                return best

        # Did not hit the WR target — still save the best we found so the
        # user can inspect it and resume training later.
        best_overall.metrics = {
            **best_overall.metrics,
            **{f"test_{k}": v for k, v in self._test(best_overall).items()},
        }
        best_overall.save(save_path)
        return best_overall
