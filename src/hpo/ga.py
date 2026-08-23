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


def _ga_numpy(
    fitness_func, lb: np.ndarray, ub: np.ndarray,
    sol_per_pop: int = 10, n_generations: int = 10,
    num_parents_mating: int = 4, mutation_rate: float = 0.2,
    seed: int = 0
) -> tuple[np.ndarray, float, list[float]]:
    rng = np.random.default_rng(seed)
    dim = len(lb)

    # Initial population
    pop = lb + (ub - lb) * rng.random(size=(sol_per_pop, dim))
    fitnesses = np.array([fitness_func(ind) for ind in pop], dtype=float)

    best_idx = np.argmax(fitnesses)
    best_ind = pop[best_idx].copy()
    best_fit = float(fitnesses[best_idx])
    history = [-best_fit]  # Convert back to error for history tracking

    for _ in range(n_generations):
        # Selection: Tournament selection
        parents = []
        for _ in range(num_parents_mating):
            tourn_idx = rng.choice(sol_per_pop, size=3, replace=False)
            winner = tourn_idx[np.argmax(fitnesses[tourn_idx])]
            parents.append(pop[winner])
        parents = np.array(parents)

        # Crossover & Mutation to create new population
        next_pop = [best_ind.copy()]  # Elitism

        while len(next_pop) < sol_per_pop:
            p1_idx, p2_idx = rng.choice(len(parents), size=2, replace=False)
            p1, p2 = parents[p1_idx], parents[p2_idx]

            # Uniform / single-point crossover
            cross_pt = rng.integers(1, dim)
            child = np.concatenate([p1[:cross_pt], p2[cross_pt:]])

            # Mutation
            for d in range(dim):
                if rng.random() < mutation_rate:
                    child[d] = lb[d] + (ub[d] - lb[d]) * rng.random()

            child = np.clip(child, lb, ub)
            next_pop.append(child)

        pop = np.array(next_pop)
        fitnesses = np.array([fitness_func(ind) for ind in pop], dtype=float)

        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > best_fit:
            best_fit = float(fitnesses[best_idx])
            best_ind = pop[best_idx].copy()

        history.append(-best_fit)

    return best_ind, best_fit, history


def run_ga(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    n_generations: int = 10,
    sol_per_pop: int = 10,
    num_parents_mating: int = 4,
    n_steps: int = 1200,
) -> dict[str, Any]:
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()

    def fitness_func_raw(solution):
        cfg = _decode_solution(np.asarray(solution), space, base)
        metrics = train_pinn(cfg)
        rel_l2 = float(metrics["val_rel_l2"])
        return -rel_l2

    try:
        import pygad

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

        def pygad_fitness(ga_instance, solution, solution_idx):
            return fitness_func_raw(solution)

        ga = pygad.GA(
            num_generations=int(n_generations),
            num_parents_mating=int(num_parents_mating),
            sol_per_pop=int(sol_per_pop),
            num_genes=len(gene_space),
            gene_space=gene_space,
            fitness_func=pygad_fitness,
            random_seed=int(seed),
            parent_selection_type="sss",
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=20,
        )
        ga.run()
        solution, solution_fitness, _ = ga.best_solution()
        history = [-float(f) for f in getattr(ga, "best_solutions_fitness", [solution_fitness])]
    except ImportError:
        solution, solution_fitness, history = _ga_numpy(
            fitness_func_raw, lb, ub,
            sol_per_pop=int(sol_per_pop), n_generations=int(n_generations),
            num_parents_mating=int(num_parents_mating), seed=seed
        )

    best_cfg = _decode_solution(np.asarray(solution), space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["optimizer_name"] = "GA"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/ga_best_metrics.json", best_metrics)
    return best_metrics
