"""Hybrid ACO-GA Optimization Algorithm for Hyperparameter Tuning.

Uses Ant Colony continuous archive (ACOR) for initial global pheromone exploration,
then injects elite candidates into Genetic Algorithm for schema recombination & fine-tuning.
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


def run_hybrid_aco_ga(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    pop_size: int = 12,
    aco_iterations: int = 4,
    ga_generations: int = 4,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Run Hybrid ACO-GA algorithm."""
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()
    dim = len(lb)

    def objective(x: np.ndarray) -> float:
        cfg = decode_solution(x, space, base)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    # ----------------------------------------------------
    # Phase 1: ACO Archive Exploration
    # ----------------------------------------------------
    archive_size = max(10, pop_size)
    zeta = 0.85
    q = 0.5

    A = lb + (ub - lb) * rng.random(size=(archive_size, dim))
    f = np.array([objective(x) for x in A], dtype=float)

    best_f = float(np.min(f))
    history = [best_f]

    for _ in range(aco_iterations):
        order = np.argsort(f)
        A = A[order]
        f = f[order]

        k_idx = np.arange(archive_size)
        w = (1.0 / (q * archive_size * np.sqrt(2.0 * np.pi))) * np.exp(
            - (k_idx ** 2) / (2.0 * (q * archive_size) ** 2)
        )
        w = w / np.sum(w)

        sigma = np.zeros(dim, dtype=float)
        for d in range(dim):
            diff = np.abs(A[:, d] - np.dot(w, A[:, d]))
            sigma[d] = zeta * np.mean(diff) + 1e-8

        new_X = np.zeros((pop_size, dim), dtype=float)
        new_f = np.zeros(pop_size, dtype=float)
        for i in range(pop_size):
            x_new = np.zeros(dim, dtype=float)
            for d in range(dim):
                idx = rng.choice(archive_size, p=w)
                mean = A[idx, d]
                s = sigma[d]
                val = rng.normal(loc=mean, scale=s)
                val = max(lb[d], min(ub[d], val))
                x_new[d] = val
            new_X[i] = x_new
            new_f[i] = objective(x_new)

        A = np.vstack([A, new_X])
        f = np.concatenate([f, new_f])
        order = np.argsort(f)
        A = A[order][:archive_size]
        f = f[order][:archive_size]

        best_f = float(f[0])
        history.append(best_f)

    # ----------------------------------------------------
    # Phase 2: Seed GA Population from Top ACO Solutions
    # ----------------------------------------------------
    pop = A[:pop_size].copy()
    fitnesses = f[:pop_size].copy()
    best_ind = pop[0].copy()

    # ----------------------------------------------------
    # Phase 3: GA Evolutionary Fine-Tuning
    # ----------------------------------------------------
    for _ in range(ga_generations):
        parents = []
        for _ in range(max(2, pop_size // 3)):
            tourn_idx = rng.choice(pop_size, size=3, replace=False)
            winner = tourn_idx[np.argmin(fitnesses[tourn_idx])]
            parents.append(pop[winner])
        parents = np.array(parents)

        next_pop = [best_ind.copy()]  # Elitism

        while len(next_pop) < pop_size:
            p1_idx, p2_idx = rng.choice(len(parents), size=2, replace=False)
            p1, p2 = parents[p1_idx], parents[p2_idx]

            cross_pt = rng.integers(1, dim)
            child = np.concatenate([p1[:cross_pt], p2[cross_pt:]])

            for d in range(dim):
                if rng.random() < 0.15:
                    child[d] = lb[d] + (ub[d] - lb[d]) * rng.random()

            child = np.clip(child, lb, ub)
            next_pop.append(child)

        pop = np.array(next_pop)
        fitnesses = np.array([objective(ind) for ind in pop], dtype=float)

        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_f:
            best_f = float(fitnesses[best_idx])
            best_ind = pop[best_idx].copy()

        history.append(best_f)

    best_cfg = decode_solution(best_ind, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["optimizer_name"] = "ACO-GA Hybrid"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/hybrid_aco_ga_best_metrics.json", best_metrics)
    return best_metrics
