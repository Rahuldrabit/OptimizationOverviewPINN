"""Tests verifying that all 13 HPO algorithms properly record and return population diversity trajectories."""

import sys
from pathlib import Path
import pytest
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from src.hpo.comparison import ALGORITHM_REGISTRY


ALL_13_ALGORITHMS = [
    "GA", "PSO", "ACO", "GSA",
    "Fuzzy-GA", "Fuzzy-PSO", "Fuzzy-ACO",
    "GA-PSO Hybrid", "PSO-GSA Hybrid", "ACO-GA Hybrid",
    "F-MAGSO (Novel)",
    "PDE-Robust-DE",
    "Two-Stage Evo (Buzaev 2026)",
]


@pytest.mark.parametrize("alg_name", ALL_13_ALGORITHMS)
def test_algorithm_returns_valid_diversity_history(tmp_path, alg_name):
    """Verify that every algorithm returns non-empty diversity_history with values in [0, 1]."""
    runner = ALGORITHM_REGISTRY[alg_name]
    out_dir = str(tmp_path / alg_name.lower().replace(" ", "_"))

    # Run quick smoke test (steps=10 for speed)
    metrics = runner(out_dir, "ode", seed=42, steps=10, quick=True)

    # 1. Diversity history must exist and not be empty
    assert "diversity_history" in metrics, f"{alg_name} missing 'diversity_history' key in metrics"
    div_hist = metrics["diversity_history"]
    assert isinstance(div_hist, list), f"{alg_name} 'diversity_history' must be a list"
    assert len(div_hist) > 0, f"{alg_name} 'diversity_history' must not be empty"

    # 2. Every entry must have a valid diversity score in [0.0, 1.0]
    for step in div_hist:
        assert "diversity" in step, f"{alg_name} step missing 'diversity' key: {step}"
        d_val = step["diversity"]
        assert isinstance(d_val, (float, np.floating, int)), f"{alg_name} diversity must be a float, got {type(d_val)}"
        assert not np.isnan(d_val), f"{alg_name} diversity must not be NaN"
        assert 0.0 <= d_val <= 1.0 + 1e-6, f"{alg_name} diversity must be in [0, 1], got {d_val}"

    # 3. Iteration history must exist and not be empty
    assert "history" in metrics, f"{alg_name} missing 'history' key"
    assert len(metrics["history"]) > 0, f"{alg_name} 'history' must not be empty"
    for err in metrics["history"]:
        assert not np.isnan(err), f"{alg_name} history contains NaN: {metrics['history']}"
