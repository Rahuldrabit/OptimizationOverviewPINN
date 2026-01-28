from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..training.pinn_trainer import TrainConfig, train_pinn
from .search_space import SearchSpace, choose_activation, choose_optimizer, clip_float, clip_int


def _decode_position(x: np.ndarray, space: SearchSpace, base: TrainConfig) -> TrainConfig:
    """Decode a continuous vector to a TrainConfig.

    x: [layers, width, act_idx, opt_idx, log10_lr, w_phys, w_ic, n_collocation]
    """

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


def run_aco(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    n_ants: int = 10,
    n_iterations: int = 10,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Simple Ant Colony Optimization (continuous) for hyperparameters.

    Uses an ACOR-style archive of solutions; minimizes validation relative L2 error.
    """

    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)

    # Bounds for continuous representation.
    lb = np.array(
        [
            space.hidden_layers_min,
            space.hidden_width_min,
            0,
            0,
            np.log10(space.lr_min),
            space.w_phys_min,
            space.w_ic_min,
            space.n_collocation_min,
        ],
        dtype=float,
    )
    ub = np.array(
        [
            space.hidden_layers_max,
            space.hidden_width_max,
            len(space.activations) - 1,
            len(space.optimizers) - 1,
            np.log10(space.lr_max),
            space.w_phys_max,
            space.w_ic_max,
            space.n_collocation_max,
        ],
        dtype=float,
    )

    dim = lb.size

    def sample_uniform(n: int) -> np.ndarray:
        return lb + (ub - lb) * rng.random(size=(n, dim))

    def objective(x: np.ndarray) -> float:
        cfg = _decode_position(x, space, base)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    # Archive-based ACOR parameters.
    archive_size = max(10, n_ants)
    zeta = 0.85  # exploration vs exploitation
    q = 0.5

    # Initialize archive uniformly.
    A = sample_uniform(archive_size)
    f = np.array([objective(x) for x in A], dtype=float)

    for _ in range(int(n_iterations)):
        # Sort archive by fitness (lower is better).
        order = np.argsort(f)
        A = A[order]
        f = f[order]

        # Compute weights.
        k_idx = np.arange(archive_size)
        # Gaussian kernel centered on best solution index 0.
        w = (1.0 / (q * archive_size * np.sqrt(2.0 * np.pi))) * np.exp(
            - (k_idx ** 2) / (2.0 * (q * archive_size) ** 2)
        )
        w = w / np.sum(w)

        # Standard deviations per dimension.
        sigma = np.zeros(dim, dtype=float)
        for d in range(dim):
            diff = np.abs(A[:, d] - np.dot(w, A[:, d]))
            sigma[d] = zeta * np.mean(diff) + 1e-8

        # Generate new ants.
        new_X = np.zeros((n_ants, dim), dtype=float)
        new_f = np.zeros(n_ants, dtype=float)
        for i in range(n_ants):
            x_new = np.zeros(dim, dtype=float)
            for d in range(dim):
                # Select archive index using weights.
                idx = rng.choice(archive_size, p=w)
                mean = A[idx, d]
                s = sigma[d]
                val = rng.normal(loc=mean, scale=s)
                # Clamp to bounds.
                val = max(lb[d], min(ub[d], val))
                x_new[d] = val
            new_X[i] = x_new
            new_f[i] = objective(x_new)

        # Merge and truncate archive.
        A = np.vstack([A, new_X])
        f = np.concatenate([f, new_f])
        order = np.argsort(f)
        A = A[order][:archive_size]
        f = f[order][:archive_size]

    best_x = A[0]
    best_cfg = _decode_position(best_x, space, base)
    best_metrics = train_pinn(best_cfg)

    from ..utils import save_json

    save_json(f"{out_dir}/aco_best_metrics.json", best_metrics)
    return best_metrics