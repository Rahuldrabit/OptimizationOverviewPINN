from __future__ import annotations

from dataclasses import replace
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



def _decode_position(x: np.ndarray, space: SearchSpace, base: TrainConfig) -> TrainConfig:
    return decode_solution(x, space, base)


def run_gsa(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    n_agents: int = 12,
    n_iterations: int = 10,
    G0: float = 100.0,
    alpha: float = 20.0,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Gravitational Search Algorithm (GSA) for PINN Hyperparameter Optimization.

    Based on Rashedi et al. (2009). Agents interact via gravitational forces proportional
    to their fitness masses, balancing exploration (high G) and exploitation (low G).
    """
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()
    dim = len(lb)
    eps = 1e-8

    def objective(x: np.ndarray) -> float:
        cfg = _decode_position(x, space, base)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    # Initialize agent positions and velocities
    X = lb + (ub - lb) * rng.random(size=(n_agents, dim))
    V = np.zeros((n_agents, dim), dtype=float)

    # Evaluate initial population
    fitness = np.array([objective(x) for x in X], dtype=float)

    best_idx = np.argmin(fitness)
    best_x = X[best_idx].copy()
    best_f = float(fitness[best_idx])
    history = [best_f]

    for it in range(1, int(n_iterations) + 1):
        # Gravitational constant decay
        G = G0 * np.exp(-alpha * float(it) / float(n_iterations))

        best_fit = np.min(fitness)
        worst_fit = np.max(fitness)

        if np.isclose(best_fit, worst_fit):
            mass = np.ones(n_agents) / float(n_agents)
        else:
            # For minimization: lower fitness means higher mass
            q = (worst_fit - fitness) / (worst_fit - best_fit + eps)
            mass = q / (np.sum(q) + eps)

        # kbest: in early iterations all agents exert force; later only top agents
        kbest_count = int(np.ceil(n_agents * (1.0 - 0.7 * (it / float(n_iterations)))))
        kbest_count = max(1, min(n_agents, kbest_count))
        sorted_indices = np.argsort(fitness)
        kbest_indices = sorted_indices[:kbest_count]

        # Calculate gravitational acceleration on each agent
        acc = np.zeros((n_agents, dim), dtype=float)

        for i in range(n_agents):
            force_i = np.zeros(dim, dtype=float)
            for j in kbest_indices:
                if i != j:
                    # Euclidean distance in normalized space
                    norm_diff = (X[j] - X[i]) / (ub - lb + eps)
                    R = np.linalg.norm(norm_diff) + eps
                    rand_j = rng.random(size=dim)
                    force_i += rand_j * G * mass[j] * (X[j] - X[i]) / R
            acc[i] = force_i

        # Velocity and position update
        r = rng.random(size=(n_agents, dim))
        V = r * V + acc
        # Velocity clamping
        v_max = (ub - lb) * 0.25
        V = np.clip(V, -v_max, v_max)

        X = np.clip(X + V, lb, ub)

        # Evaluate new positions
        for i in range(n_agents):
            fit = objective(X[i])
            fitness[i] = fit
            if fit < best_f:
                best_f = float(fit)
                best_x = X[i].copy()

        history.append(best_f)

    best_cfg = _decode_position(best_x, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["optimizer_name"] = "GSA"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/gsa_best_metrics.json", best_metrics)
    return best_metrics

