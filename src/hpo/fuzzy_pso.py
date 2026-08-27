"""Fuzzy-Adaptive Particle Swarm Optimization (Fuzzy-PSO).

Uses a Mamdani Fuzzy Logic Controller to dynamically adapt inertia weight and
acceleration coefficients based on swarm diversity and search progression.
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


def run_fuzzy_pso(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    swarmsize: int = 12,
    maxiter: int = 8,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Run Fuzzy-Adaptive PSO optimization for PINN hyperparameters."""
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()
    dim = len(lb)
    v_max = (ub - lb) * 0.25

    flc = FuzzyController()

    def objective(x: np.ndarray) -> float:
        cfg = decode_solution(x, space, base)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    # Initialize swarm positions and velocities
    X = lb + (ub - lb) * rng.random(size=(swarmsize, dim))
    V = -v_max + 2 * v_max * rng.random(size=(swarmsize, dim))

    # Evaluate initial swarm
    P = X.copy()
    P_fit = np.array([objective(x) for x in X], dtype=float)

    best_idx = np.argmin(P_fit)
    gbest = P[best_idx].copy()
    gbest_fit = float(P_fit[best_idx])
    history = [gbest_fit]
    prev_gbest_fit = gbest_fit

    fuzzy_adaptations = []

    for it in range(1, int(maxiter) + 1):
        # Compute fuzzy metrics
        diversity = compute_population_diversity(X, lb, ub)
        improvement = float(max(0.0, (prev_gbest_fit - gbest_fit) / (prev_gbest_fit + 1e-12)))
        progress = float(it / maxiter)

        explore_w, exploit_w = flc.evaluate(diversity, improvement, progress)

        # Adapt parameters via fuzzy rules
        w = 0.3 + 0.6 * explore_w
        c2 = 1.0 + 1.5 * exploit_w
        c1 = float(np.clip(3.2 - c2, 0.8, 2.5))

        fuzzy_adaptations.append({
            "iteration": it,
            "diversity": diversity,
            "improvement": improvement,
            "w": w,
            "c1": c1,
            "c2": c2,
        })

        prev_gbest_fit = gbest_fit

        # Particle updates
        r1 = rng.random(size=(swarmsize, dim))
        r2 = rng.random(size=(swarmsize, dim))

        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, lb, ub)

        for i in range(swarmsize):
            fit = objective(X[i])
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
    best_metrics["fuzzy_adaptations"] = fuzzy_adaptations
    best_metrics["diversity_history"] = [
        {"iteration": a["iteration"], "diversity": a["diversity"]} for a in fuzzy_adaptations
    ]
    best_metrics["optimizer_name"] = "Fuzzy-PSO"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/fuzzy_pso_best_metrics.json", best_metrics)
    return best_metrics
