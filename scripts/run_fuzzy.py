"""CLI runner for Fuzzy-Enhanced HPO optimizers (Fuzzy-PSO, Fuzzy-GA, Fuzzy-ACO)."""

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

from hpo.fuzzy_pso import run_fuzzy_pso
from hpo.fuzzy_ga import run_fuzzy_ga
from hpo.fuzzy_aco import run_fuzzy_aco
from utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fuzzy-Adaptive Hyperparameter Optimization")
    parser.add_argument("benchmark", nargs="?", default="ode", help="Benchmark PDE type (default: ode)")
    parser.add_argument("--method", choices=["pso", "ga", "aco", "all"], default="all", help="Fuzzy optimizer to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=1200, help="Training steps per candidate")
    parser.add_argument("--quick", action="store_true", help="Run quick test with reduced iterations")

    args = parser.parse_args()
    benchmark_type = args.benchmark
    n_iters = 2 if args.quick else 8

    methods_to_run = ["pso", "ga", "aco"] if args.method == "all" else [args.method]

    print(f"{'='*60}")
    print(f"Running Fuzzy-Enhanced HPO on '{benchmark_type}' benchmark")
    print(f"Methods: {', '.join(m.upper() for m in methods_to_run)} | Seed: {args.seed} | Quick: {args.quick}")
    print(f"{'='*60}")

    for m in methods_to_run:
        out_dir = os.path.join("outputs", f"fuzzy_{m}", benchmark_type)
        ensure_dir(out_dir)

        print(f"\n---> Executing Fuzzy-{m.upper()}...")
        if m == "pso":
            metrics = run_fuzzy_pso(
                out_dir=out_dir,
                benchmark_type=benchmark_type,
                seed=args.seed,
                swarmsize=6 if args.quick else 12,
                maxiter=n_iters,
                n_steps=args.steps,
            )
        elif m == "ga":
            metrics = run_fuzzy_ga(
                out_dir=out_dir,
                benchmark_type=benchmark_type,
                seed=args.seed,
                n_generations=n_iters,
                sol_per_pop=6 if args.quick else 10,
                num_parents_mating=2 if args.quick else 4,
                n_steps=args.steps,
            )
        elif m == "aco":
            metrics = run_fuzzy_aco(
                out_dir=out_dir,
                benchmark_type=benchmark_type,
                seed=args.seed,
                n_ants=6 if args.quick else 10,
                n_iterations=n_iters,
                n_steps=args.steps,
            )

        print(f"Results for Fuzzy-{m.upper()}:")
        print(f"  val_rel_l2 = {metrics['val_rel_l2']:.6f} | val_mse = {metrics['val_mse']:.6e}")
        print(f"  Best Config: layers={metrics['config']['hidden_layers']}, width={metrics['config']['hidden_width']}, act={metrics['config']['activation']}, opt={metrics['config']['optimizer']}, lr={metrics['config']['lr']:.2e}")


if __name__ == "__main__":
    main()
