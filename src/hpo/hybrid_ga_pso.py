"""Hybrid GA-PSO Optimization Algorithm for Hyperparameter Tuning.

Alternates between Genetic Algorithm exploration (crossover & mutation) and
Particle Swarm Optimization exploitation (velocity & memory guidance).
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


def run_hybrid_ga_pso(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    pop_size: int = 12,
    n_epochs: int = 4,
    ga_gens_per_epoch: int = 2,
    pso_iters_per_epoch: int = 2,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Run Hybrid GA-PSO algorithm."""
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()
    dim = len(lb)
    v_max = (ub - lb) * 0.2

    def objective(x: np.ndarray) -> float:
        cfg = decode_solution(x, space, base)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    # Initialize shared population / swarm
    X = lb + (ub - lb) * rng.random(size=(pop_size, dim))
    V = -v_max + 2 * v_max * rng.random(size=(pop_size, dim))
    P = X.copy()
    fitness = np.array([objective(x) for x in X], dtype=float)
    P_fit = fitness.copy()

    best_idx = np.argmin(fitness)
    gbest = X[best_idx].copy()
    gbest_fit = float(fitness[best_idx])
    history = [gbest_fit]

    for epoch in range(n_epochs):
        # ----------------------------------------------------
        # Stage 1: GA Evolution (Exploration Phase)
        # ----------------------------------------------------
        for _ in range(ga_gens_per_epoch):
            # Tournament selection
            parents = []
            for _ in range(max(2, pop_size // 3)):
                tourn_idx = rng.choice(pop_size, size=3, replace=False)
                winner = tourn_idx[np.argmin(fitness[tourn_idx])]
                parents.append(X[winner])
            parents = np.array(parents)

            new_pop = [gbest.copy()]  # Elitism

            while len(new_pop) < pop_size:
                p1_idx, p2_idx = rng.choice(len(parents), size=2, replace=False)
                p1, p2 = parents[p1_idx], parents[p2_idx]

                cross_pt = rng.integers(1, dim)
                child = np.concatenate([p1[:cross_pt], p2[cross_pt:]])

                # Mutation
                for d in range(dim):
                    if rng.random() < 0.2:
                        child[d] = lb[d] + (ub[d] - lb[d]) * rng.random()

                child = np.clip(child, lb, ub)
                new_pop.append(child)

            X = np.array(new_pop)
            for i in range(pop_size):
                fit = objective(X[i])
                fitness[i] = fit
                if fit < P_fit[i]:
                    P_fit[i] = fit
                    P[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = float(fit)
                    gbest = X[i].copy()

            history.append(gbest_fit)

        # ----------------------------------------------------
        # Stage 2: PSO Velocity Update (Exploitation Phase)
        # ----------------------------------------------------
        w = 0.6
        c1 = 1.4
        c2 = 1.6
        for _ in range(pso_iters_per_epoch):
            r1 = rng.random(size=(pop_size, dim))
            r2 = rng.random(size=(pop_size, dim))

            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (gbest - X)
            V = np.clip(V, -v_max, v_max)
            X = np.clip(X + V, lb, ub)

            for i in range(pop_size):
                fit = objective(X[i])
                fitness[i] = fit
                if fit < P_fit[i]:
                    P_fit[i] = fit
                    P[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = float(fit)
                    gbest = X[i].copy()

            history.append(gbest_fit)

    best_cfg = decode_solution(gbest, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["optimizer_name"] = "GA-PSO Hybrid"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/hybrid_ga_pso_best_metrics.json", best_metrics)
    return best_metrics
