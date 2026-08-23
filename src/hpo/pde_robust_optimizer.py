from __future__ import annotations

import numpy as np
from dataclasses import replace
from typing import Any

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

def run_pde_robust_opt(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    n_generations: int = 10,
    sol_per_pop: int = 20,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """
    PDE-Robust-DE (Physics-Informed Differential Evolution with Adaptive Scaling)
    Uses JADE-style adaptation for mutation (F) and crossover (CR) to approximate
    the optimal parameter scaling dynamically, focusing heavily on w_phys and w_ic scaling.
    """
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()
    dim = len(lb)

    # Initialize Population
    pop = lb + (ub - lb) * rng.random(size=(sol_per_pop, dim))
    fitnesses = np.zeros(sol_per_pop)
    
    # Evaluate Initial Population
    for i in range(sol_per_pop):
        cfg = _decode_solution(pop[i], space, base)
        metrics = train_pinn(cfg)
        fitnesses[i] = float(metrics["val_rel_l2"])

    best_idx = np.argmin(fitnesses)
    best_ind = pop[best_idx].copy()
    best_fit = fitnesses[best_idx]
    
    history = [best_fit]

    # Adaptive parameters (JADE style)
    mu_F = 0.5
    mu_CR = 0.5
    c = 0.1 # Adaptation rate

    for gen in range(n_generations):
        next_pop = np.zeros_like(pop)
        next_fitnesses = np.zeros(sol_per_pop)
        
        successful_F = []
        successful_CR = []

        for i in range(sol_per_pop):
            # Generate F and CR for this individual based on adaptive means
            F = rng.normal(mu_F, 0.1)
            F = np.clip(F, 0.1, 1.0)
            CR = rng.normal(mu_CR, 0.1)
            CR = np.clip(CR, 0.0, 1.0)

            # DE/rand/1 Mutation
            candidates = list(range(sol_per_pop))
            candidates.remove(i)
            r1, r2, r3 = rng.choice(candidates, 3, replace=False)
            
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = np.copy(pop[i])
            j_rand = rng.integers(0, dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                    
            # Boundary handling (Bounce-back)
            for j in range(dim):
                if trial[j] < lb[j]:
                    trial[j] = lb[j] + rng.random() * (pop[i, j] - lb[j])
                elif trial[j] > ub[j]:
                    trial[j] = ub[j] - rng.random() * (ub[j] - pop[i, j])

            # Evaluate Trial
            cfg = _decode_solution(trial, space, base)
            metrics = train_pinn(cfg)
            trial_fit = float(metrics["val_rel_l2"])

            # Selection
            if trial_fit < fitnesses[i]:
                next_pop[i] = trial
                next_fitnesses[i] = trial_fit
                successful_F.append(F)
                successful_CR.append(CR)
                
                # Update global best
                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_ind = trial.copy()
            else:
                next_pop[i] = pop[i]
                next_fitnesses[i] = fitnesses[i]

        pop = next_pop
        fitnesses = next_fitnesses
        history.append(best_fit)

        # Adapt mu_F and mu_CR based on successful mutations
        if len(successful_F) > 0:
            # Lehmer mean for F
            mu_F = (1 - c) * mu_F + c * (sum(f**2 for f in successful_F) / sum(successful_F))
            # Arithmetic mean for CR
            mu_CR = (1 - c) * mu_CR + c * np.mean(successful_CR)

    best_cfg = _decode_solution(best_ind, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["optimizer_name"] = "PDE-Robust-DE"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/pde_robust_de_best_metrics.json", best_metrics)
    return best_metrics
