from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from hpo.gsa import run_gsa
from utils import ensure_dir


def main() -> None:
    """Run Gravitational Search Optimization for PINN hyperparameters."""
    benchmark_type = sys.argv[1] if len(sys.argv) > 1 else "ode"

    out_dir = os.path.join("outputs", "gsa", benchmark_type)
    ensure_dir(out_dir)

    print(f"Running GSA optimization on {benchmark_type} benchmark...")
    metrics = run_gsa(
        out_dir=out_dir,
        benchmark_type=benchmark_type,
        seed=0,
        n_agents=12,
        n_iterations=8,
        G0=100.0,
        alpha=20.0,
        n_steps=1200,
    )

    print(f"\nGSA Results for {benchmark_type}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
