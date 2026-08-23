"""Fuzzy-Adaptive Genetic Algorithm (Fuzzy-GA).

Uses a Mamdani Fuzzy Logic Controller to dynamically adjust mutation rate,
crossover probability, and selection pressure based on population diversity.
"""

from __future__ import annotations

from typing import Any
import numpy as np

try:
    from ..training.pinn_trainer import TrainConfig, train_pinn
    from ..utils import ensure_dir, save_json
    from .search_space import (
        SearchSpace,
        choose_activation,
        choose_optimizer,
        clip_float,
        clip_int,
        decode_solution,
    )
    from .fuzzy_controller import FuzzyController, compute_population_diversity
except (ImportError, ValueError):
    from training.pinn_trainer import TrainConfig, train_pinn
    from utils import ensure_dir, save_json
    from hpo.search_space import (
        SearchSpace,
        choose_activation,
        choose_optimizer,
        clip_float,
        clip_int,
        decode_solution,
    )
    from hpo.fuzzy_controller import FuzzyController, compute_population_diversity


def run_fuzzy_ga(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    n_generations: int = 10,
    sol_per_pop: int = 10,
    num_parents_mating: int = 4,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Run Fuzzy-Adaptive GA optimization for PINN hyperparameters."""
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()
    dim = len(lb)

    flc = FuzzyController()

    def objective(x: np.ndarray) -> float:
        cfg = decode_solution(x, space, base)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    # Initial population
    pop = lb + (ub - lb) * rng.random(size=(sol_per_pop, dim))
    fitnesses = np.array([objective(ind) for ind in pop], dtype=float)

    best_idx = np.argmin(fitnesses)
    best_ind = pop[best_idx].copy()
    best_fit = float(fitnesses[best_idx])
    history = [best_fit]
    prev_best_fit = best_fit

    fuzzy_adaptations = []

    for gen in range(1, int(n_generations) + 1):
        diversity = compute_population_diversity(pop, lb, ub)
        improvement = float(max(0.0, (prev_best_fit - best_fit) / (prev_best_fit + 1e-12)))
        progress = float(gen / n_generations)

        explore_w, exploit_w = flc.evaluate(diversity, improvement, progress)

        # Adapt GA hyperparameters
        mutation_rate = float(np.clip(0.05 + 0.35 * explore_w, 0.05, 0.45))
        crossover_prob = float(np.clip(0.50 + 0.45 * exploit_w, 0.50, 0.95))

        fuzzy_adaptations.append({
            "generation": gen,
            "diversity": diversity,
            "improvement": improvement,
            "mutation_rate": mutation_rate,
            "crossover_prob": crossover_prob,
        })

        prev_best_fit = best_fit

        # Selection: Tournament
        parents = []
        for _ in range(num_parents_mating):
            tourn_idx = rng.choice(sol_per_pop, size=3, replace=False)
            winner = tourn_idx[np.argmin(fitnesses[tourn_idx])]
            parents.append(pop[winner])
        parents = np.array(parents)

        next_pop = [best_ind.copy()]  # Elitism

        while len(next_pop) < sol_per_pop:
            p1_idx, p2_idx = rng.choice(len(parents), size=2, replace=False)
            p1, p2 = parents[p1_idx], parents[p2_idx]

            # Crossover with probability
            if rng.random() < crossover_prob:
                cross_pt = rng.integers(1, dim)
                child = np.concatenate([p1[:cross_pt], p2[cross_pt:]])
            else:
                child = p1.copy()

            # Adaptive mutation
            for d in range(dim):
                if rng.random() < mutation_rate:
                    # Adaptive mutation step: larger when exploring, smaller when exploiting
                    step = (ub[d] - lb[d]) * (0.1 + 0.4 * explore_w)
                    child[d] += rng.normal(0.0, step)

            child = np.clip(child, lb, ub)
            next_pop.append(child)

        pop = np.array(next_pop)
        fitnesses = np.array([objective(ind) for ind in pop], dtype=float)

        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_fit:
            best_fit = float(fitnesses[best_idx])
            best_ind = pop[best_idx].copy()

        history.append(best_fit)

    best_cfg = decode_solution(best_ind, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["fuzzy_adaptations"] = fuzzy_adaptations
    best_metrics["optimizer_name"] = "Fuzzy-GA"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/fuzzy_ga_best_metrics.json", best_metrics)
    return best_metrics
