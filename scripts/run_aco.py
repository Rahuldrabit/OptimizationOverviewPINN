from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpo.aco import run_aco
from utils import ensure_dir


def main() -> None:
    """Run ant colony optimization for hyperparameters."""
    benchmark_type = sys.argv[1] if len(sys.argv) > 1 else "ode"
    
    out_dir = os.path.join("outputs", "aco", benchmark_type)
    ensure_dir(out_dir)

    print(f"Running ACO optimization on {benchmark_type} benchmark...")
    metrics = run_aco(
        out_dir=out_dir, 
        benchmark_type=benchmark_type,
        seed=0, 
        n_ants=10, 
        n_iterations=8, 
        n_steps=1200
    )
    
    print(f"\nACO Results for {benchmark_type}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
