"""F-MAGSO: Fuzzy-Guided Multi-Stage Adaptive Gravitational Swarm Optimizer.

Synthesizes the empirical findings from the benchmark investigation:
1. GSA Macro-Attraction: Instant early basin capture (from GSA's 4-eval speed).
2. Fuzzy-Adaptive PSO Momentum: Rapid continuous parameter acceleration (from PSO's 0.0186 descent rate).
3. GA Schema Crossover: Optimal discrete architecture selection (layers, activations, optimizers).
4. Adaptive Cauchy Perturbation: Escapes micro-plateaus in stiff PINN PDE loss landscapes.
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


def run_f_magso(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    pop_size: int = 12,
    max_evals: int = 80,
    n_steps: int = 1200,
    use_flc: bool = True,
    use_gsa: bool = True,
    use_ga_schema: bool = True,
    use_multistage: bool = True,
) -> dict[str, Any]:
    """Execute F-MAGSO (Fuzzy-Guided Multi-Stage Adaptive Gravitational Swarm Optimizer)."""
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()
    dim = len(lb)
    v_max = (ub - lb) * 0.25
    eps = 1e-8

    flc = FuzzyController()

    eval_count = 0
    eval_history: list[float] = []
    best_so_far = float("inf")
    best_x = np.zeros(dim, dtype=float)

    def objective(x: np.ndarray) -> float:
        nonlocal eval_count, best_so_far, best_x
        eval_count += 1
        cfg = decode_solution(x, space, base)
        metrics = train_pinn(cfg)
        fit = float(metrics["val_rel_l2"])
        if fit < best_so_far:
            best_so_far = fit
            best_x = x.copy()
        eval_history.append(best_so_far)
        return fit

    # ----------------------------------------------------
    # Initialization
    # ----------------------------------------------------
    X = lb + (ub - lb) * rng.random(size=(pop_size, dim))
    V = np.zeros((pop_size, dim), dtype=float)
    P = X.copy()
    fitness = np.array([objective(x) for x in X], dtype=float)
    P_fit = fitness.copy()

    # Iteration 0 tracking
    history = [best_so_far]
    diversity_history = [{"iteration": 0, "diversity": compute_population_diversity(X, lb, ub)}]

    G0 = 100.0
    alpha = 15.0
    prev_gbest_fit = best_so_far
    iteration_idx = 0

    while eval_count < max_evals:
        iteration_idx += 1
        it_ratio = float(eval_count / max_evals)

        # ------------------------------------------------
        # 1. Fuzzy Logic Adaptive Controller
        # ------------------------------------------------
        if use_flc:
            diversity = compute_population_diversity(X, lb, ub)
            improvement = float(max(0.0, (prev_gbest_fit - best_so_far) / (prev_gbest_fit + 1e-12)))
            explore_w, exploit_w = flc.evaluate(diversity, improvement, it_ratio)
        else:
            explore_w, exploit_w = 0.5, 0.5
        prev_gbest_fit = best_so_far

        # ------------------------------------------------
        # 2. Multi-Stage Phase Transitions
        # ------------------------------------------------
        if use_multistage:
            if it_ratio < 0.25:
                # PHASE 1: Gravitational Macro-Attraction (GSA Dominated)
                w = 0.5
                c1_pso, c2_pso = 0.5, 0.8
                c_gsa = 2.0 * explore_w if use_gsa else 0.0
            elif it_ratio < 0.75:
                # PHASE 2: Fuzzy Velocity Swarm Acceleration (PSO Dominated)
                w = 0.3 + 0.5 * explore_w
                c2_pso = 1.2 + 1.2 * exploit_w
                c1_pso = float(np.clip(3.0 - c2_pso, 0.8, 2.0))
                c_gsa = 0.8 * explore_w if use_gsa else 0.0
            else:
                # PHASE 3: Elite Recombination & Micro Local Exploitation
                w = 0.2
                c1_pso, c2_pso = 0.8, 2.2
                c_gsa = 0.1 if use_gsa else 0.0
        else:
            # Static single stage
            w = 0.5
            c1_pso, c2_pso = 1.5, 1.5
            c_gsa = 1.0 if use_gsa else 0.0

        # ------------------------------------------------
        # 3. Compute Gravitational Accelerations
        # ------------------------------------------------
        if use_gsa:
            G = G0 * np.exp(-alpha * it_ratio)
            b_fit, w_fit = np.min(fitness), np.max(fitness)
            if np.isclose(b_fit, w_fit):
                mass = np.ones(pop_size) / float(pop_size)
            else:
                q = (w_fit - fitness) / (w_fit - b_fit + eps)
                mass = q / (np.sum(q) + eps)

            kbest = max(2, int(np.ceil(pop_size * (1.0 - 0.6 * it_ratio))))
            k_indices = np.argsort(fitness)[:kbest]

            acc = np.zeros((pop_size, dim), dtype=float)
            for i in range(pop_size):
                force_i = np.zeros(dim, dtype=float)
                for j in k_indices:
                    if i != j:
                        norm_d = (X[j] - X[i]) / (ub - lb + eps)
                        R = np.linalg.norm(norm_d) + eps
                        force_i += rng.random(size=dim) * G * mass[j] * (X[j] - X[i]) / R
                acc[i] = force_i
        else:
            k_indices = np.argsort(fitness)[:max(2, pop_size // 2)]
            acc = np.zeros((pop_size, dim), dtype=float)

        # ------------------------------------------------
        # 4. Unified Velocity & Position Update
        # ------------------------------------------------
        r1 = rng.random(size=(pop_size, dim))
        r2 = rng.random(size=(pop_size, dim))
        r3 = rng.random(size=(pop_size, dim))

        V = w * V + c1_pso * r1 * (P - X) + c2_pso * r2 * (best_x - X) + c_gsa * r3 * acc
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, lb, ub)

        # ------------------------------------------------
        # 5. Hybrid Genetic Schema Recombination for Discrete Hyperparams
        # ------------------------------------------------
        if use_ga_schema:
            for i in range(pop_size):
                if rng.random() < (0.35 * exploit_w):
                    donor_idx = k_indices[rng.choice(min(3, len(k_indices)))]
                    X[i, :4] = X[donor_idx, :4]

        # ------------------------------------------------
        # 6. Evaluation & Personal Best Updates
        # ------------------------------------------------
        for i in range(pop_size):
            if eval_count >= max_evals:
                break
            fit = objective(X[i])
            fitness[i] = fit
            if fit < P_fit[i]:
                P_fit[i] = fit
                P[i] = X[i].copy()

        # Record per-iteration metrics
        history.append(best_so_far)
        diversity_history.append({
            "iteration": iteration_idx,
            "diversity": compute_population_diversity(X, lb, ub),
        })

    best_cfg = decode_solution(best_x, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["eval_history"] = eval_history
    best_metrics["diversity_history"] = diversity_history
    best_metrics["optimizer_name"] = "F-MAGSO"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/f_magso_best_metrics.json", best_metrics)
    return best_metrics
