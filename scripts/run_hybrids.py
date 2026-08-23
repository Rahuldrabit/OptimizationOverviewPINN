"""CLI runner for Hybrid HPO optimizers (GA-PSO, PSO-GSA, ACO-GA)."""

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

from hpo.hybrid_ga_pso import run_hybrid_ga_pso
from hpo.hybrid_pso_gsa import run_hybrid_pso_gsa
from hpo.hybrid_aco_ga import run_hybrid_aco_ga
from utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hybrid Hyperparameter Optimization")
    parser.add_argument("benchmark", nargs="?", default="ode", help="Benchmark PDE type (default: ode)")
    parser.add_argument("--method", choices=["ga_pso", "pso_gsa", "aco_ga", "all"], default="all", help="Hybrid optimizer to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=1200, help="Training steps per candidate")
    parser.add_argument("--quick", action="store_true", help="Run quick test with reduced budget")

    args = parser.parse_args()
    benchmark_type = args.benchmark

    methods_to_run = ["ga_pso", "pso_gsa", "aco_ga"] if args.method == "all" else [args.method]

    print(f"{'='*60}")
    print(f"Running Hybrid HPO on '{benchmark_type}' benchmark")
    print(f"Methods: {', '.join(m.upper() for m in methods_to_run)} | Seed: {args.seed} | Quick: {args.quick}")
    print(f"{'='*60}")

    for m in methods_to_run:
        out_dir = os.path.join("outputs", f"hybrid_{m}", benchmark_type)
        ensure_dir(out_dir)

        print(f"\n---> Executing Hybrid {m.upper()}...")
        if m == "ga_pso":
            metrics = run_hybrid_ga_pso(
                out_dir=out_dir,
                benchmark_type=benchmark_type,
                seed=args.seed,
                pop_size=6 if args.quick else 12,
                n_epochs=2 if args.quick else 4,
                ga_gens_per_epoch=1 if args.quick else 2,
                pso_iters_per_epoch=1 if args.quick else 2,
                n_steps=args.steps,
            )
        elif m == "pso_gsa":
            metrics = run_hybrid_pso_gsa(
                out_dir=out_dir,
                benchmark_type=benchmark_type,
                seed=args.seed,
                n_agents=6 if args.quick else 12,
                n_iterations=3 if args.quick else 8,
                n_steps=args.steps,
            )
        elif m == "aco_ga":
            metrics = run_hybrid_aco_ga(
                out_dir=out_dir,
                benchmark_type=benchmark_type,
                seed=args.seed,
                pop_size=6 if args.quick else 12,
                aco_iterations=2 if args.quick else 4,
                ga_generations=2 if args.quick else 4,
                n_steps=args.steps,
            )

        print(f"Results for Hybrid {m.upper()}:")
        print(f"  val_rel_l2 = {metrics['val_rel_l2']:.6f} | val_mse = {metrics['val_mse']:.6e}")
        print(f"  Best Config: layers={metrics['config']['hidden_layers']}, width={metrics['config']['hidden_width']}, act={metrics['config']['activation']}, opt={metrics['config']['optimizer']}, lr={metrics['config']['lr']:.2e}")


if __name__ == "__main__":
    main()
