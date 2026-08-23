"""CLI runner for the Novel F-MAGSO (Fuzzy-Guided Multi-Stage Adaptive Gravitational Swarm Optimizer)."""

from __future__ import annotations

import argparse
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

from hpo.novel_f_magso import run_f_magso
from utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Novel F-MAGSO for PINN Hyperparameter Optimization")
    parser.add_argument("benchmark", nargs="?", default="ode", help="Benchmark PDE type (default: ode)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--evals", type=int, default=60, help="Max evaluations budget")
    parser.add_argument("--steps", type=int, default=1200, help="Training steps per candidate")

    args = parser.parse_args()
    out_dir = os.path.join("outputs", "f_magso", args.benchmark)
    ensure_dir(out_dir)

    print(f"\n{'='*70}")
    print(f"RUNNING NOVEL F-MAGSO OPTIMIZER ON '{args.benchmark.upper()}' BENCHMARK")
    print(f"Max Evaluations: {args.evals} | Seed: {args.seed} | PINN Steps: {args.steps}")
    print(f"{'='*70}\n")

    metrics = run_f_magso(
        out_dir=out_dir,
        benchmark_type=args.benchmark,
        seed=args.seed,
        max_evals=args.evals,
        n_steps=args.steps,
    )

    print(f"F-MAGSO Results for {args.benchmark.upper()}:")
    print(f"  Final Val Rel L2 = {metrics['val_rel_l2']:.6e}")
    print(f"  Final Val MSE    = {metrics['val_mse']:.6e}")
    print(f"  Best Config:")
    print(json.dumps(metrics["config"], indent=4))
    print(f"\nDetailed output saved to: {out_dir}/f_magso_best_metrics.json\n")


if __name__ == "__main__":
    main()
