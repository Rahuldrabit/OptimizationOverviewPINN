from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpo.ga import run_ga
from utils import ensure_dir


def main() -> None:
    """Run genetic algorithm hyperparameter optimization."""
    benchmark_type = sys.argv[1] if len(sys.argv) > 1 else "ode"
    
    out_dir = os.path.join("outputs", "ga", benchmark_type)
    ensure_dir(out_dir)

    print(f"Running GA optimization on {benchmark_type} benchmark...")
    metrics = run_ga(
        out_dir=out_dir, 
        benchmark_type=benchmark_type,
        seed=0, 
        n_generations=8, 
        sol_per_pop=10, 
        num_parents_mating=4, 
        n_steps=1200
    )
    
    print(f"\nGA Results for {benchmark_type}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
