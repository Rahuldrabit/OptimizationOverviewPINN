"""Unit test for DEAP Genetic Algorithm PINN validation."""

import pytest
import numpy as np
from src.hpo.deap_ga import run_deap_ga, run_deap_ga_pinn


def test_deap_ga_sphere_convergence():
    """Test DEAP GA on a simple sphere test function."""
    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])

    def sphere_fn(x):
        return float(np.sum(x ** 2))

    best_x, best_fit, history = run_deap_ga(
        lb, ub, sphere_fn, pop_size=10, n_generations=5, seed=42
    )

    assert len(best_x) == 2
    assert best_fit >= 0.0
    assert len(history) == 6  # gen 0 + 5 gens
    assert best_fit < 5.0


def test_deap_ga_pinn_smoke(tmp_path):
    """Smoke test DEAP GA on ODE PINN benchmark."""
    res = run_deap_ga_pinn(
        out_dir=str(tmp_path),
        benchmark_type="ode",
        seed=0,
        pop_size=4,
        n_generations=2,
        n_steps=100,
    )

    assert "val_rel_l2" in res
    assert "history" in res
    assert res["val_rel_l2"] > 0.0
