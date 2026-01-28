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


def _decode_solution(solution: np.ndarray, space: SearchSpace, base: TrainConfig) -> TrainConfig:
    # Genes: [layers, width, act_idx, opt_idx, log10_lr, w_phys, w_ic, n_collocation]
    layers = clip_int(solution[0], space.hidden_layers_min, space.hidden_layers_max)
    width = clip_int(solution[1], space.hidden_width_min, space.hidden_width_max)
    activation = choose_activation(solution[2], space.activations)
    optimizer = choose_optimizer(solution[3], space.optimizers)

    log10_lr = clip_float(solution[4], np.log10(space.lr_min), np.log10(space.lr_max))
    lr = float(10 ** log10_lr)

    w_phys = clip_float(solution[5], space.w_phys_min, space.w_phys_max)
    w_ic = clip_float(solution[6], space.w_ic_min, space.w_ic_max)

    n_col = clip_int(solution[7], space.n_collocation_min, space.n_collocation_max)

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


def run_ga(
    out_dir: str,
    seed: int = 0,
    n_generations: int = 10,
    sol_per_pop: int = 10,
    num_parents_mating: int = 4,
    n_steps: int = 1200,
) -> dict[str, Any]:
    import pygad

    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps)

    gene_space = [
        {"low": space.hidden_layers_min, "high": space.hidden_layers_max},
        {"low": space.hidden_width_min, "high": space.hidden_width_max},
        {"low": 0, "high": len(space.activations) - 1},
        {"low": 0, "high": len(space.optimizers) - 1},
        {"low": float(np.log10(space.lr_min)), "high": float(np.log10(space.lr_max))},
        {"low": space.w_phys_min, "high": space.w_phys_max},
        {"low": space.w_ic_min, "high": space.w_ic_max},
        {"low": space.n_collocation_min, "high": space.n_collocation_max},
    ]

    def fitness_func(ga_instance, solution, solution_idx):
        cfg = _decode_solution(np.asarray(solution), space, base)
        metrics = train_pinn(cfg)
        # Maximize fitness; we want low error.
        rel_l2 = float(metrics["val_rel_l2"])
        fitness = -rel_l2
        return fitness

    ga = pygad.GA(
        num_generations=int(n_generations),
        num_parents_mating=int(num_parents_mating),
        sol_per_pop=int(sol_per_pop),
        num_genes=len(gene_space),
        gene_space=gene_space,
        fitness_func=fitness_func,
        random_seed=int(seed),
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20,
    )

    ga.run()

    solution, solution_fitness, _ = ga.best_solution()
    best_cfg = _decode_solution(np.asarray(solution), space, base)
    best_metrics = train_pinn(best_cfg)

    from .utils import save_json

    save_json(f"{out_dir}/ga_best_metrics.json", best_metrics)
    return best_metrics
