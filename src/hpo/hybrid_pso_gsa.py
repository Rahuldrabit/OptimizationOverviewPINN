"""Hybrid PSO-GSA Optimization Algorithm for Hyperparameter Tuning.

Combines Particle Swarm Optimization (social exploitation) with Gravitational
Search Algorithm (mass interaction exploration) in a unified velocity equation.
Based on Mirjalili & Hashim (2010).
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


def run_hybrid_pso_gsa(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    n_agents: int = 12,
    n_iterations: int = 8,
    G0: float = 100.0,
    alpha: float = 20.0,
    c1_prime: float = 1.0,  # Gravitational acceleration weight
    c2_prime: float = 1.5,  # PSO gbest attraction weight
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Run Hybrid PSO-GSA algorithm."""
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()
    dim = len(lb)
    v_max = (ub - lb) * 0.25
    eps = 1e-8

    def objective(x: np.ndarray) -> float:
        cfg = decode_solution(x, space, base)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    # Initialize positions and velocities
    X = lb + (ub - lb) * rng.random(size=(n_agents, dim))
    V = np.zeros((n_agents, dim), dtype=float)

    fitness = np.array([objective(x) for x in X], dtype=float)

    best_idx = np.argmin(fitness)
    gbest = X[best_idx].copy()
    gbest_fit = float(fitness[best_idx])
    history = [gbest_fit]

    for it in range(1, int(n_iterations) + 1):
        # 1. Decay G(t) and compute masses
        G = G0 * np.exp(-alpha * float(it) / float(n_iterations))
        best_fit = np.min(fitness)
        worst_fit = np.max(fitness)

        if np.isclose(best_fit, worst_fit):
            mass = np.ones(n_agents) / float(n_agents)
        else:
            q = (worst_fit - fitness) / (worst_fit - best_fit + eps)
            mass = q / (np.sum(q) + eps)

        # 2. Compute GSA acceleration
        kbest_count = int(np.ceil(n_agents * (1.0 - 0.7 * (it / float(n_iterations)))))
        kbest_count = max(1, min(n_agents, kbest_count))
        sorted_indices = np.argsort(fitness)
        kbest_indices = sorted_indices[:kbest_count]

        acc = np.zeros((n_agents, dim), dtype=float)
        for i in range(n_agents):
            force_i = np.zeros(dim, dtype=float)
            for j in kbest_indices:
                if i != j:
                    norm_diff = (X[j] - X[i]) / (ub - lb + eps)
                    R = np.linalg.norm(norm_diff) + eps
                    rand_j = rng.random(size=dim)
                    force_i += rand_j * G * mass[j] * (X[j] - X[i]) / R
            acc[i] = force_i

        # 3. Hybrid Velocity Update: Inertia + GSA acceleration + PSO gbest pull
        w = 0.9 - 0.5 * (it / float(n_iterations))  # Linearly decreasing inertia
        r1 = rng.random(size=(n_agents, dim))
        r2 = rng.random(size=(n_agents, dim))

        V = w * V + c1_prime * r1 * acc + c2_prime * r2 * (gbest - X)
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, lb, ub)

        # 4. Fitness evaluation and global best update
        for i in range(n_agents):
            fit = objective(X[i])
            fitness[i] = fit
            if fit < gbest_fit:
                gbest_fit = float(fit)
                gbest = X[i].copy()

        history.append(gbest_fit)

    best_cfg = decode_solution(gbest, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["optimizer_name"] = "PSO-GSA Hybrid"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/hybrid_pso_gsa_best_metrics.json", best_metrics)
    return best_metrics
