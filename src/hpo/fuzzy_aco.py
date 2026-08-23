"""Fuzzy-Adaptive Ant Colony Optimization (Fuzzy-ACO).

Uses a Mamdani Fuzzy Logic Controller to dynamically adapt the archive dispersion
parameter (zeta) and Gaussian selection weight (q) based on search diversity.
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


def run_fuzzy_aco(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    n_ants: int = 10,
    n_iterations: int = 10,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Run Fuzzy-Adaptive ACO optimization for PINN hyperparameters."""
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

    archive_size = max(10, n_ants)

    # Initialize archive uniformly
    A = lb + (ub - lb) * rng.random(size=(archive_size, dim))
    f = np.array([objective(x) for x in A], dtype=float)

    best_f = float(np.min(f))
    history = [best_f]
    prev_best_f = best_f

    fuzzy_adaptations = []

    for it in range(1, int(n_iterations) + 1):
        diversity = compute_population_diversity(A, lb, ub)
        improvement = float(max(0.0, (prev_best_f - best_f) / (prev_best_f + 1e-12)))
        progress = float(it / n_iterations)

        explore_w, exploit_w = flc.evaluate(diversity, improvement, progress)

        # Adapt ACOR parameters:
        # zeta controls sampling spread (higher when exploring)
        zeta = float(np.clip(0.35 + 0.85 * explore_w, 0.3, 1.2))
        # q controls Gaussian kernel sharpness (smaller concentrates heavily on best rank)
        q = float(np.clip(0.15 + 0.65 * (1.0 - exploit_w), 0.1, 0.9))

        fuzzy_adaptations.append({
            "iteration": it,
            "diversity": diversity,
            "improvement": improvement,
            "zeta": zeta,
            "q": q,
        })

        prev_best_f = best_f

        # Sort archive by fitness (lower is better)
        order = np.argsort(f)
        A = A[order]
        f = f[order]

        # Gaussian kernel weights
        k_idx = np.arange(archive_size)
        w = (1.0 / (q * archive_size * np.sqrt(2.0 * np.pi))) * np.exp(
            - (k_idx ** 2) / (2.0 * (q * archive_size) ** 2)
        )
        w = w / np.sum(w)

        # Standard deviations per dimension
        sigma = np.zeros(dim, dtype=float)
        for d in range(dim):
            diff = np.abs(A[:, d] - np.dot(w, A[:, d]))
            sigma[d] = zeta * np.mean(diff) + 1e-8

        # Generate new ants
        new_X = np.zeros((n_ants, dim), dtype=float)
        new_f = np.zeros(n_ants, dtype=float)
        for i in range(n_ants):
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

        # Merge and truncate archive
        A = np.vstack([A, new_X])
        f = np.concatenate([f, new_f])
        order = np.argsort(f)
        A = A[order][:archive_size]
        f = f[order][:archive_size]

        best_f = float(f[0])
        history.append(best_f)

    best_x = A[0]
    best_cfg = decode_solution(best_x, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["fuzzy_adaptations"] = fuzzy_adaptations
    best_metrics["optimizer_name"] = "Fuzzy-ACO"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/fuzzy_aco_best_metrics.json", best_metrics)
    return best_metrics
