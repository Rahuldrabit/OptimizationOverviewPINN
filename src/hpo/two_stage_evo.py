"""Two-Stage Evolutionary Strategy for PINNs (Buzaev et al., 2026).

Paper: "Evolutionary Two-Stage Hyperparameter Optimization Strategies for
Physics-Informed Neural Networks" (arXiv:2606.20442 / ICLR 2026 Workshop).

Architecture:
- Stage 1 (Screening / Exploration): Evolutionary Algorithm (DE/GA) with coarse
  evaluation budget (70% of total evals) to identify promising candidate basins.
- Stage 2 (Refinement / Exploitation): Fine local neighbourhood exploitation on the top-K
  candidates using localized perturbation / gradient-oriented refinement.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np

try:
    from ..utils import ensure_dir, save_json
    from ..training.pinn_trainer import TrainConfig, train_pinn
    from .search_space import SearchSpace, decode_solution
except (ImportError, ValueError):
    from utils import ensure_dir, save_json
    from training.pinn_trainer import TrainConfig, train_pinn
    from hpo.search_space import SearchSpace, decode_solution


def _two_stage_evo_numpy(
    lb: np.ndarray,
    ub: np.ndarray,
    eval_func: Callable[[np.ndarray], float],
    max_evals: int,
    stage1_ratio: float = 0.70,
    pop_size: int = 15,
    top_k: int = 3,
    mutation_factor: float = 0.5,
    crossover_rate: float = 0.7,
    seed: int = 42,
) -> tuple[np.ndarray, float, list[float]]:
    """Execute Two-Stage Evolutionary Strategy under fixed evaluation budget."""
    rng = np.random.default_rng(seed)
    dim = len(lb)
    stage1_eval_budget = int(max_evals * stage1_ratio)
    history: list[float] = []

    # -------------------------------------------------------------
    # STAGE 1: Coarse Evolutionary Exploration (Differential Evolution)
    # -------------------------------------------------------------
    pop = lb + (ub - lb) * rng.random(size=(pop_size, dim))
    fitness = np.zeros(pop_size)
    eval_count = 0

    for i in range(pop_size):
        if eval_count >= stage1_eval_budget or eval_count >= max_evals:
            break
        fit = eval_func(pop[i])
        fitness[i] = fit
        eval_count += 1
        history.append(float(np.min(fitness[:eval_count])))

    best_idx = int(np.argmin(fitness[:max(1, eval_count)]))
    best_x = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    while eval_count < stage1_eval_budget and eval_count < max_evals:
        for i in range(pop_size):
            if eval_count >= stage1_eval_budget or eval_count >= max_evals:
                break

            candidates = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = rng.choice(candidates, 3, replace=False)

            # DE/rand/1 mutation
            mutant = pop[r1] + mutation_factor * (pop[r2] - pop[r3])

            # Binomial crossover
            trial = np.copy(pop[i])
            j_rand = rng.integers(0, dim)
            for j in range(dim):
                if rng.random() < crossover_rate or j == j_rand:
                    trial[j] = mutant[j]

            # Boundary constraint handling
            trial = np.clip(trial, lb, ub)

            fit = eval_func(trial)
            eval_count += 1

            if fit <= fitness[i]:
                pop[i] = trial
                fitness[i] = fit
                if fit < best_fit:
                    best_fit = fit
                    best_x = trial.copy()

            history.append(float(best_fit))

    # -------------------------------------------------------------
    # STAGE 2: Fine Refinement / Exploitation on Top-K Elite Candidates
    # -------------------------------------------------------------
    ranked_indices = np.argsort(fitness)
    elite_pool = [pop[idx].copy() for idx in ranked_indices[:top_k]]
    elite_fitness = [fitness[idx] for idx in ranked_indices[:top_k]]

    refine_step = 0.05 * (ub - lb)

    while eval_count < max_evals:
        for k in range(min(top_k, len(elite_pool))):
            if eval_count >= max_evals:
                break

            direction = rng.normal(0, 1, size=dim)
            perturbed = elite_pool[k] + direction * refine_step * (0.8 ** (eval_count / max(1, max_evals)))
            perturbed = np.clip(perturbed, lb, ub)

            fit = eval_func(perturbed)
            eval_count += 1

            if fit < elite_fitness[k]:
                elite_pool[k] = perturbed.copy()
                elite_fitness[k] = fit
                if fit < best_fit:
                    best_fit = fit
                    best_x = perturbed.copy()

            history.append(float(best_fit))

    return best_x, float(best_fit), history


def run_two_stage_evo(*args: Any, **kwargs: Any) -> Any:
    """Flexible runner interface for Two-Stage Evo supporting both comparison grid and speed benchmark."""
    # Caller style 1: run_two_stage_evo(tracker, space, max_evals=..., seed=...)
    if len(args) >= 2 and hasattr(args[0], "evaluate"):
        tracker = args[0]
        space = args[1]
        max_evals = kwargs.get("max_evals", 60)
        seed = kwargs.get("seed", 42)
        lb, ub = space.get_bounds()
        best_x, best_fit, _ = _two_stage_evo_numpy(lb, ub, tracker.evaluate, max_evals, seed=seed)
        return best_x, best_fit

    # Caller style 2: run_two_stage_evo(output_dir, benchmark_type, seed=..., max_evals=..., n_steps=...)
    out_dir = args[0] if len(args) > 0 else kwargs.get("output_dir", "outputs/comparison")
    benchmark_type = args[1] if len(args) > 1 else kwargs.get("benchmark_type", "ode")
    seed = kwargs.get("seed", 0)
    max_evals = kwargs.get("max_evals", 80)
    n_steps = kwargs.get("n_steps", 1200)

    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()

    def eval_fn(cand: np.ndarray) -> float:
        cfg = decode_solution(cand, space, base)
        m = train_pinn(cfg)
        return float(m["val_rel_l2"])

    best_x, best_fit, history = _two_stage_evo_numpy(lb, ub, eval_fn, max_evals=max_evals, seed=seed)

    best_cfg = decode_solution(best_x, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["optimizer_name"] = "Two-Stage Evo (Buzaev 2026)"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/two_stage_evo_best_metrics.json", best_metrics)
    return best_metrics
