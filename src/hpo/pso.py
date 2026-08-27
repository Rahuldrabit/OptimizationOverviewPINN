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
    from .fuzzy_controller import compute_population_diversity
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
    from hpo.fuzzy_controller import compute_population_diversity



def _decode_position(x: np.ndarray, space: SearchSpace, base: TrainConfig) -> TrainConfig:
    # x: [layers, width, act_idx, opt_idx, log10_lr, w_phys, w_ic, n_collocation]
    layers = clip_int(x[0], space.hidden_layers_min, space.hidden_layers_max)
    width = clip_int(x[1], space.hidden_width_min, space.hidden_width_max)
    activation = choose_activation(x[2], space.activations)
    optimizer = choose_optimizer(x[3], space.optimizers)

    log10_lr = clip_float(x[4], np.log10(space.lr_min), np.log10(space.lr_max))
    lr = float(10 ** log10_lr)

    w_phys = clip_float(x[5], space.w_phys_min, space.w_phys_max)
    w_ic = clip_float(x[6], space.w_ic_min, space.w_ic_max)

    n_col = clip_int(x[7], space.n_collocation_min, space.n_collocation_max)

    return replace(
        base,
        hidden_layers=layers,
        hidden_width=width,
        activation=activation,
        optimizer=optimizer,
        lr=lr,
        w_phys=w_phys,
        w_ic=w_ic,
        n_collocation=n_col,
    )


def _pso_numpy(
    func, lb: np.ndarray, ub: np.ndarray, swarmsize: int = 12, maxiter: int = 8,
    w: float = 0.7, c1: float = 1.5, c2: float = 1.5, seed: int = 0
) -> tuple[np.ndarray, float, list[float], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    dim = len(lb)
    v_max = (ub - lb) * 0.2

    # Initialize positions and velocities
    X = lb + (ub - lb) * rng.random(size=(swarmsize, dim))
    V = -v_max + 2 * v_max * rng.random(size=(swarmsize, dim))

    # Evaluate initial swarm
    P = X.copy()
    P_fit = np.array([func(x) for x in X], dtype=float)

    best_idx = np.argmin(P_fit)
    gbest = P[best_idx].copy()
    gbest_fit = float(P_fit[best_idx])
    history = [gbest_fit]
    diversity_history = [{"iteration": 0, "diversity": compute_population_diversity(X, lb, ub)}]

    for it in range(1, int(maxiter) + 1):
        r1 = rng.random(size=(swarmsize, dim))
        r2 = rng.random(size=(swarmsize, dim))

        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, lb, ub)

        for i in range(swarmsize):
            fit = func(X[i])
            if fit < P_fit[i]:
                P_fit[i] = fit
                P[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = float(fit)
                    gbest = X[i].copy()

        history.append(gbest_fit)
        diversity_history.append({"iteration": it, "diversity": compute_population_diversity(X, lb, ub)})

    return gbest, gbest_fit, history, diversity_history


def run_pso(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    swarmsize: int = 12,
    maxiter: int = 8,
    n_steps: int = 1200,
) -> dict[str, Any]:
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()

    def objective(x):
        cfg = _decode_position(np.asarray(x), space, base)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    try:
        from pyswarm import pso
        best_x, best_f = pso(objective, lb, ub, swarmsize=int(swarmsize), maxiter=int(maxiter))
        history = [float(best_f)]
        diversity_history: list[dict[str, Any]] = []  # pyswarm 0.6 exposes no per-iteration swarm hook
    except ImportError:
        best_x, best_f, history, diversity_history = _pso_numpy(objective, lb, ub, swarmsize=int(swarmsize), maxiter=int(maxiter), seed=seed)

    best_cfg = _decode_position(np.asarray(best_x), space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["diversity_history"] = diversity_history
    best_metrics["optimizer_name"] = "PSO"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/pso_best_metrics.json", best_metrics)
    return best_metrics
