from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpo.pso import run_pso
from utils import ensure_dir


def main() -> None:
    """Run particle swarm optimization for hyperparameters."""
    benchmark_type = sys.argv[1] if len(sys.argv) > 1 else "ode"
    
    out_dir = os.path.join("outputs", "pso", benchmark_type)
    ensure_dir(out_dir)

    print(f"Running PSO optimization on {benchmark_type} benchmark...")
    metrics = run_pso(
        out_dir=out_dir, 
        benchmark_type=benchmark_type,
        seed=0, 
        swarmsize=12, 
        maxiter=6, 
        n_steps=1200
    )
    
    print(f"\nPSO Results for {benchmark_type}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
