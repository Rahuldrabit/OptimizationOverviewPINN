"""OmniPINN-Opt: Unified Multi-Specialty Optimizer for Physics-Informed Neural Networks.

Synthesizes the 5 empirical metaheuristic strengths discovered during benchmarking:
1. GSA Macro-Gravitational Pull: Ultra-fast early-stage basin identification (GSA speed).
2. PSO Velocity Momentum: Directional swarm exploitation & steep descent slope (PSO speed).
3. ACO Continuous Gaussian Pheromone Archive: Deep anti-trapping landscape smoothing (ACO precision).
4. GA Discrete Schema Crossover & Pheromone Routing: Robust neural architecture search (GA rank #1).
5. Mamdani Fuzzy Logic Controller (FLC 2.0): Dynamic real-time force blending and parameter self-tuning.
"""

from __future__ import annotations

import os
import time
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


class MamdaniFLC2:
    """Enhanced Dual-Input Multi-Output Mamdani Fuzzy Inference Engine."""

    def __init__(self) -> None:
        self.u_out = np.linspace(0.0, 1.0, 101)

    @staticmethod
    def _trimf(x: float, abc: tuple[float, float, float]) -> float:
        a, b, c = abc
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:
            return (c - x) / (c - b) if c != b else 1.0

    def evaluate(
        self,
        diversity: float,
        improvement_rate: float,
        progress: float
    ) -> dict[str, float]:
        """Infer dynamic control parameters.
        
        Returns:
            Dictionary containing explore_weight, momentum_weight, exploit_weight,
            mutation_prob, and aco_kernel_scale.
        """
        d = float(np.clip(diversity, 0.0, 1.0))
        imp = float(np.clip(improvement_rate, 0.0, 1.0))
        prog = float(np.clip(progress, 0.0, 1.0))

        # Membership evaluations
        mu_d_low = self._trimf(d, (0.0, 0.0, 0.45))
        mu_d_med = self._trimf(d, (0.2, 0.5, 0.8))
        mu_d_high = self._trimf(d, (0.55, 1.0, 1.0))

        mu_imp_stag = self._trimf(imp, (0.0, 0.0, 0.3))
        mu_imp_fast = self._trimf(imp, (0.4, 1.0, 1.0))

        mu_prog_early = self._trimf(prog, (0.0, 0.0, 0.45))
        mu_prog_mid = self._trimf(prog, (0.25, 0.5, 0.75))
        mu_prog_late = self._trimf(prog, (0.55, 1.0, 1.0))

        # Dynamic weights
        explore_w = float(np.clip(
            0.6 * mu_prog_early + 0.4 * mu_d_high + 0.3 * mu_imp_stag,
            0.1, 1.0
        ))
        momentum_w = float(np.clip(
            0.7 * mu_prog_mid + 0.5 * mu_imp_fast + 0.3 * mu_d_med,
            0.1, 1.0
        ))
        exploit_w = float(np.clip(
            0.8 * mu_prog_late + 0.4 * (1.0 - mu_d_high) + 0.3 * (1.0 - mu_imp_stag),
            0.1, 1.0
        ))

        # GA Mutation probability: increases when diversity is critically low or stagnating
        mutation_prob = float(np.clip(0.05 + 0.25 * (1.0 - d) * mu_imp_stag + 0.15 * explore_w, 0.02, 0.40))
        
        # ACO Kernel scale: narrows during late exploitation
        kernel_scale = float(np.clip(0.8 * explore_w + 0.2 * (1.0 - exploit_w), 0.05, 1.0))

        return {
            "explore_w": explore_w,
            "momentum_w": momentum_w,
            "exploit_w": exploit_w,
            "mutation_prob": mutation_prob,
            "kernel_scale": kernel_scale,
        }


def run_omnipinn_opt(
    out_dir: str,
    benchmark_type: str = "ode",
    seed: int = 0,
    pop_size: int = 12,
    max_evals: int = 80,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Execute the Unified OmniPINN-Opt Metaheuristic."""
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()
    dim = len(lb)
    eps = 1e-8

    # Continuous parameter indices: 4 (log10_lr), 5 (w_phys), 6 (w_ic), 7 (n_collocation)
    cont_idx = np.array([4, 5, 6, 7])
    disc_idx = np.array([0, 1, 2, 3])
    lb_c, ub_c = lb[cont_idx], ub[cont_idx]
    dim_c = len(cont_idx)
    v_max_c = (ub_c - lb_c) * 0.25

    flc = MamdaniFLC2()

    eval_count = 0
    history: list[float] = []
    best_so_far = float("inf")
    best_x = np.zeros(dim, dtype=float)

    # Historical Solution Archive for ACOR continuous sampling & GA discrete schema
    archive_size = min(15, pop_size * 2)
    archive_X: list[np.ndarray] = []
    archive_fits: list[float] = []

    def objective(x: np.ndarray) -> float:
        nonlocal eval_count, best_so_far, best_x, archive_X, archive_fits
        eval_count += 1
        cfg = decode_solution(x, space, base)
        metrics = train_pinn(cfg)
        fit = float(metrics["val_rel_l2"])
        
        if fit < best_so_far:
            best_so_far = fit
            best_x = x.copy()
            
        history.append(best_so_far)

        # Update Solution Archive
        archive_X.append(x.copy())
        archive_fits.append(fit)
        if len(archive_fits) > archive_size:
            sort_idx = np.argsort(archive_fits)
            archive_X = [archive_X[i] for i in sort_idx[:archive_size]]
            archive_fits = [archive_fits[i] for i in sort_idx[:archive_size]]

        return fit

    # ----------------------------------------------------
    # Initialization
    # ----------------------------------------------------
    X = lb + (ub - lb) * rng.random(size=(pop_size, dim))
    V_c = np.zeros((pop_size, dim_c), dtype=float)
    P = X.copy()
    fitness = np.array([objective(x) for x in X], dtype=float)
    P_fit = fitness.copy()

    # Discrete Pheromone Table for categorical decisions
    tau_act = np.ones(len(space.activations), dtype=float)
    tau_opt = np.ones(len(space.optimizers), dtype=float)

    G0 = 100.0
    alpha_gsa = 12.0
    prev_gbest_fit = best_so_far

    while eval_count < max_evals:
        it_ratio = float(eval_count / max_evals)

        # ------------------------------------------------
        # 1. Fuzzy Logic Controller Dynamic Sensing
        # ------------------------------------------------
        diversity = compute_population_diversity(X, lb, ub)
        improvement = float(max(0.0, (prev_gbest_fit - best_so_far) / (prev_gbest_fit + 1e-12)))
        f_params = flc.evaluate(diversity, improvement, it_ratio)
        prev_gbest_fit = best_so_far

        exp_w = f_params["explore_w"]
        mom_w = f_params["momentum_w"]
        expt_w = f_params["exploit_w"]
        p_mut = f_params["mutation_prob"]
        k_scale = f_params["kernel_scale"]

        # Smooth Dynamic Force Blending Functions (eliminates rigid phase cuts)
        # Early GSA dominance -> Mid PSO momentum -> Late ACOR Gaussian Polish
        alpha_t = 2.0 * np.exp(-4.0 * it_ratio) * exp_w
        beta_t = 4.0 * it_ratio * (1.0 - it_ratio) * (1.2 + 0.8 * mom_w)
        gamma_t = 1.8 * (it_ratio ** 2) * (0.8 + 1.2 * expt_w)

        # ------------------------------------------------
        # 2. Continuous Subspace Force Computations
        # ------------------------------------------------
        # A. GSA Gravitational Acceleration
        G = G0 * np.exp(-alpha_gsa * it_ratio)
        b_fit, w_fit = np.min(fitness), np.max(fitness)
        if np.isclose(b_fit, w_fit):
            mass = np.ones(pop_size) / float(pop_size)
        else:
            q = (w_fit - fitness) / (w_fit - b_fit + eps)
            mass = q / (np.sum(q) + eps)

        kbest = max(2, int(np.ceil(pop_size * (1.0 - 0.7 * it_ratio))))
        k_indices = np.argsort(fitness)[:kbest]

        f_gsa = np.zeros((pop_size, dim_c), dtype=float)
        for i in range(pop_size):
            force_i = np.zeros(dim_c, dtype=float)
            for j in k_indices:
                if i != j:
                    R = np.linalg.norm((X[j, cont_idx] - X[i, cont_idx]) / (ub_c - lb_c + eps)) + eps
                    force_i += rng.random(size=dim_c) * G * mass[j] * (X[j, cont_idx] - X[i, cont_idx]) / R
            f_gsa[i] = force_i

        # B. PSO Inertial & Social Force
        w_pso = 0.4 + 0.3 * exp_w - 0.2 * it_ratio
        c1_pso = 1.4 * (1.0 - 0.5 * it_ratio)
        c2_pso = 1.4 * (0.5 + 0.7 * it_ratio) * expt_w

        r1 = rng.random(size=(pop_size, dim_c))
        r2 = rng.random(size=(pop_size, dim_c))
        f_pso = w_pso * V_c + c1_pso * r1 * (P[:, cont_idx] - X[:, cont_idx]) + c2_pso * r2 * (best_x[cont_idx] - X[:, cont_idx])

        # C. ACOR Continuous Gaussian Kernel Polishing
        f_acor = np.zeros((pop_size, dim_c), dtype=float)
        if len(archive_fits) >= 2:
            n_arch = len(archive_fits)
            q_param = 0.1 + 0.3 * (1.0 - it_ratio)
            arch_weights = 1.0 / (q_param * n_arch * np.sqrt(2 * np.pi)) * np.exp(- (np.arange(n_arch) ** 2) / (2 * (q_param * n_arch) ** 2))
            arch_weights /= np.sum(arch_weights)

            arch_mat = np.array([x[cont_idx] for x in archive_X], dtype=float)
            for i in range(pop_size):
                chosen_l = rng.choice(n_arch, p=arch_weights)
                sigma_l = k_scale * np.std(arch_mat, axis=0) + 1e-4
                delta_sample = (arch_mat[chosen_l] + rng.normal(0, sigma_l, size=dim_c)) - X[i, cont_idx]
                f_acor[i] = delta_sample

        # D. Unified Force Integration
        r_g = rng.random(size=(pop_size, dim_c))
        r_a = rng.random(size=(pop_size, dim_c))
        V_c = alpha_t * r_g * f_gsa + beta_t * f_pso + gamma_t * r_a * f_acor
        V_c = np.clip(V_c, -v_max_c, v_max_c)
        X[:, cont_idx] = np.clip(X[:, cont_idx] + V_c, lb_c, ub_c)

        # ------------------------------------------------
        # 3. Discrete Subspace Dynamics (GA + ACO Pheromone Matrix)
        # ------------------------------------------------
        # Update Pheromones from elite candidates
        rho = 0.15  # Evaporation
        tau_act = (1.0 - rho) * tau_act
        tau_opt = (1.0 - rho) * tau_opt
        for k_idx in k_indices[:3]:
            act_choice = int(np.clip(np.round(X[k_idx, 2]), 0, len(space.activations) - 1))
            opt_choice = int(np.clip(np.round(X[k_idx, 3]), 0, len(space.optimizers) - 1))
            tau_act[act_choice] += 1.0 / (fitness[k_idx] + 1e-4)
            tau_opt[opt_choice] += 1.0 / (fitness[k_idx] + 1e-4)

        prob_act = tau_act / np.sum(tau_act)
        prob_opt = tau_opt / np.sum(tau_opt)

        for i in range(pop_size):
            # A. Elitist GA Crossover for architecture (layers & width)
            if rng.random() < 0.65:
                parent_a = k_indices[rng.choice(min(3, len(k_indices)))]
                parent_b = k_indices[rng.choice(len(k_indices))]
                X[i, 0] = X[parent_a, 0] if rng.random() < 0.5 else X[parent_b, 0]
                X[i, 1] = X[parent_a, 1] if rng.random() < 0.5 else X[parent_b, 1]

            # B. Pheromone-Guided Categorical Sampling (activation & optimizer)
            if rng.random() < (1.0 - p_mut):
                X[i, 2] = float(rng.choice(len(space.activations), p=prob_act))
                X[i, 3] = float(rng.choice(len(space.optimizers), p=prob_opt))
            else:
                # Random exploration mutation
                X[i, 2] = float(rng.choice(len(space.activations)))
                X[i, 3] = float(rng.choice(len(space.optimizers)))

            # C. Discrete mutation on layers & width
            if rng.random() < p_mut:
                X[i, 0] = float(rng.integers(space.hidden_layers_min, space.hidden_layers_max + 1))
            if rng.random() < p_mut:
                X[i, 1] = float(rng.integers(space.hidden_width_min, space.hidden_width_max + 1))

        # ------------------------------------------------
        # 4. PINN Candidate Evaluations & Personal Bests
        # ------------------------------------------------
        for i in range(pop_size):
            if eval_count >= max_evals:
                break
            fit = objective(X[i])
            fitness[i] = fit
            if fit < P_fit[i]:
                P_fit[i] = fit
                P[i] = X[i].copy()

    best_cfg = decode_solution(best_x, space, base)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["optimizer_name"] = "OmniPINN-Opt"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/omnipinn_opt_best_metrics.json", best_metrics)
    return best_metrics
