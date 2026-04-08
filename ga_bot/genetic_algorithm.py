"""The GA engine: population init → evaluate → select → crossover → mutate.

Stops as soon as the *validation* split's win rate is at least
`target_win_rate` AND the candidate has enough trades AND a profit factor
above the configured floor. This is the user's "stop training when WR ≥ 80%"
gate, defended against trivial 5-trade strategies.
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


@dataclass
class GenerationLog:
    generation: int
    best_fitness: float
    best_train_metrics: Dict[str, float]
    best_val_metrics: Dict[str, float]
    elapsed_sec: float


class GeneticAlgorithm:
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        on_generation: Optional[Callable[[GenerationLog], None]] = None,
    ):
        self.cfg = CONFIG.ga
        self.rng = np.random.default_rng(self.cfg.random_seed)
        self.train_bt = Backtester(train_df)
        self.val_bt = Backtester(val_df)
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
        result = self.train_bt.run(strat)
        m = evaluate(result, min_trades=30)
        c.fitness = m["fitness"]
        c.metrics = m

    def _validate(self, c: Chromosome) -> Dict[str, float]:
        result = self.val_bt.run(Strategy(c))
        return result.metrics()

    # ---------- stopping rule ----------
    def _meets_stop_criteria(self, val_metrics: Dict[str, float]) -> bool:
        return (
            val_metrics["trades"] >= self.cfg.min_trades_for_stop
            and val_metrics["win_rate"] >= self.cfg.target_win_rate
            and val_metrics["profit_factor"] >= self.cfg.min_profit_factor_for_stop
        )

    # ---------- main loop ----------
    def run(self, save_path: Optional[Path] = None) -> Chromosome:
        save_path = Path(save_path) if save_path else MODELS_DIR / "best_chromosome.json"
        pop = self._init_population()
        for c in pop:
            self._evaluate(c)
        pop.sort(key=lambda c: c.fitness, reverse=True)

        best_overall: Chromosome = pop[0]

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

            # ----- evaluate (skip already-evaluated elites) -----
            for c in new_pop[self.cfg.elite_count :]:
                self._evaluate(c)
            new_pop.sort(key=lambda c: c.fitness, reverse=True)
            pop = new_pop

            best = pop[0]
            val_metrics = self._validate(best)

            if best.fitness > best_overall.fitness:
                best_overall = best

            log = GenerationLog(
                generation=gen,
                best_fitness=best.fitness,
                best_train_metrics=best.metrics,
                best_val_metrics=val_metrics,
                elapsed_sec=time.time() - t0,
            )
            self.history.append(log)
            if self.on_generation is not None:
                self.on_generation(log)

            # ----- stop check -----
            if self._meets_stop_criteria(val_metrics):
                # Persist with both train and val metrics attached.
                best.metrics = {
                    **best.metrics,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                best.save(save_path)
                return best

        # Did not hit the WR target — still save the best we found so the
        # user can inspect it and resume training later.
        best_overall.metrics = {
            **best_overall.metrics,
            **{f"val_{k}": v for k, v in self._validate(best_overall).items()},
        }
        best_overall.save(save_path)
        return best_overall
