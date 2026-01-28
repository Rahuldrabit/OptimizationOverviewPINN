from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .pinn_ode import TrainConfig, train_pinn
from .search_space import (
    SearchSpace,
    choose_activation,
    choose_optimizer,
    clip_float,
    clip_int,
)


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


def run_pso(
    out_dir: str,
    seed: int = 0,
    swarmsize: int = 12,
    maxiter: int = 8,
    n_steps: int = 1200,
) -> dict[str, Any]:
    from pyswarm import pso

    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps)

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

    def objective(x):
        cfg = _decode_position(np.asarray(x), space, base)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    best_x, best_f = pso(objective, lb, ub, swarmsize=int(swarmsize), maxiter=int(maxiter))
    best_cfg = _decode_position(np.asarray(best_x), space, base)
    best_metrics = train_pinn(best_cfg)

    from .utils import save_json

    save_json(f"{out_dir}/pso_best_metrics.json", best_metrics)
    return best_metrics
