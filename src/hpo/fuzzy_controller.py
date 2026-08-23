"""Fuzzy Logic Controller (FLC) for Adaptive Metaheuristic Hyperparameter Optimization.

Implements a Mamdani Fuzzy Inference System in pure NumPy with triangular membership
functions, min-t-norm implication, max s-norm aggregation, and centroid defuzzification.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def _trimf(x: float | np.ndarray, abc: tuple[float, float, float]) -> float | np.ndarray:
    """Triangular membership function."""
    a, b, c = abc
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)

    # Left slope
    if b != a:
        left_mask = (a <= x_arr) & (x_arr <= b)
        y[left_mask] = (x_arr[left_mask] - a) / (b - a)
    else:
        left_mask = (x_arr == a)
        y[left_mask] = 1.0

    # Right slope
    if c != b:
        right_mask = (b <= x_arr) & (x_arr <= c)
        y[right_mask] = (c - x_arr[right_mask]) / (c - b)
    else:
        right_mask = (x_arr == c)
        y[right_mask] = 1.0

    # Peak
    y[x_arr == b] = 1.0
    y = np.clip(y, 0.0, 1.0)
    return float(y) if np.isscalar(x) else y


class FuzzyController:
    """Mamdani Fuzzy Logic Controller to dynamically balance exploration and exploitation."""

    def __init__(self) -> None:
        # Membership definitions: (a, b, c)
        # 1. Diversity (0.0 to 1.0)
        self.div_low = (0.0, 0.0, 0.45)
        self.div_med = (0.2, 0.5, 0.8)
        self.div_high = (0.55, 1.0, 1.0)

        # 2. Improvement rate (0.0 to 1.0)
        self.imp_stagnant = (0.0, 0.0, 0.3)
        self.imp_slow = (0.15, 0.5, 0.85)
        self.imp_fast = (0.6, 1.0, 1.0)

        # 3. Iteration progress t/T (0.0 to 1.0)
        self.prog_early = (0.0, 0.0, 0.45)
        self.prog_mid = (0.25, 0.5, 0.75)
        self.prog_late = (0.55, 1.0, 1.0)

        # Output universe for defuzzification (0.0 to 1.0, 101 points)
        self.u_out = np.linspace(0.0, 1.0, 101)
        self.out_low = _trimf(self.u_out, (0.0, 0.0, 0.5))
        self.out_med = _trimf(self.u_out, (0.25, 0.5, 0.75))
        self.out_high = _trimf(self.u_out, (0.5, 1.0, 1.0))

    def evaluate(
        self,
        diversity: float,
        improvement_rate: float,
        iteration_progress: float,
    ) -> tuple[float, float]:
        """Compute exploration_weight and exploitation_weight via Mamdani inference.

        Args:
            diversity: Population spread in [0, 1]
            improvement_rate: Relative fitness improvement in [0, 1]
            iteration_progress: Iteration ratio t/T in [0, 1]

        Returns:
            (exploration_weight, exploitation_weight) both in [0, 1]
        """
        # Clamp inputs
        d = float(np.clip(diversity, 0.0, 1.0))
        imp = float(np.clip(improvement_rate, 0.0, 1.0))
        prog = float(np.clip(iteration_progress, 0.0, 1.0))

        # Fuzzification
        mu_d_low = float(_trimf(d, self.div_low))
        mu_d_med = float(_trimf(d, self.div_med))
        mu_d_high = float(_trimf(d, self.div_high))

        mu_imp_stag = float(_trimf(imp, self.imp_stagnant))
        mu_imp_slow = float(_trimf(imp, self.imp_slow))
        mu_imp_fast = float(_trimf(imp, self.imp_fast))

        mu_prog_early = float(_trimf(prog, self.prog_early))
        mu_prog_mid = float(_trimf(prog, self.prog_mid))
        mu_prog_late = float(_trimf(prog, self.prog_late))

        # Rule base
        # Rule 1: IF diversity is Low AND improvement is Stagnant -> Explore High, Exploit Low
        r1 = min(mu_d_low, mu_imp_stag)
        # Rule 2: IF diversity is High AND improvement is Fast -> Explore Low, Exploit High
        r2 = min(mu_d_high, mu_imp_fast)
        # Rule 3: IF progress is Early -> Explore High, Exploit Med
        r3 = mu_prog_early
        # Rule 4: IF progress is Late AND improvement is Stagnant -> Explore Med, Exploit High
        r4 = min(mu_prog_late, mu_imp_stag)
        # Rule 5: IF progress is Late AND improvement is Fast -> Explore Low, Exploit High
        r5 = min(mu_prog_late, mu_imp_fast)
        # Rule 6: IF diversity is Med AND improvement is Slow -> Explore Med, Exploit Med
        r6 = min(mu_d_med, mu_imp_slow)
        # Rule 7: IF progress is Mid -> Explore Med, Exploit Med
        r7 = mu_prog_mid

        # Aggregation for Exploration output
        # High: r1, r3
        # Med: r4, r6, r7
        # Low: r2, r5
        exp_high = max(r1, r3)
        exp_med = max(r4, r6, r7)
        exp_low = max(r2, r5)

        agg_exp = np.maximum(
            np.minimum(exp_high, self.out_high),
            np.maximum(
                np.minimum(exp_med, self.out_med),
                np.minimum(exp_low, self.out_low)
            )
        )

        # Aggregation for Exploitation output
        # High: r2, r4, r5
        # Med: r3, r6, r7
        # Low: r1
        expt_high = max(r2, r4, r5)
        expt_med = max(r3, r6, r7)
        expt_low = r1

        agg_expt = np.maximum(
            np.minimum(expt_high, self.out_high),
            np.maximum(
                np.minimum(expt_med, self.out_med),
                np.minimum(expt_low, self.out_low)
            )
        )

        # Centroid Defuzzification
        sum_exp = np.sum(agg_exp)
        exploration = float(np.sum(self.u_out * agg_exp) / sum_exp) if sum_exp > 1e-9 else 0.5

        sum_expt = np.sum(agg_expt)
        exploitation = float(np.sum(self.u_out * agg_expt) / sum_expt) if sum_expt > 1e-9 else 0.5

        return exploration, exploitation


def compute_population_diversity(population: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    """Calculate normalized population diversity metric in [0, 1]."""
    if len(population) <= 1:
        return 0.0
    norm_pop = (population - lb) / (ub - lb + 1e-12)
    centroid = np.mean(norm_pop, axis=0)
    distances = np.linalg.norm(norm_pop - centroid, axis=1)
    # Theoretical max distance to centroid in unit hypercube with d dimensions
    max_d = np.sqrt(norm_pop.shape[1]) * 0.5
    div = float(np.mean(distances) / (max_d + 1e-12))
    return float(np.clip(div, 0.0, 1.0))
