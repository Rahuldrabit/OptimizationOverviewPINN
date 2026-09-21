"""CLI runner for PDE-Robust-DE (Physics-Informed Differential Evolution with Adaptive Scaling)."""

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

from hpo.pde_robust_optimizer import run_pde_robust_opt
from utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDE-Robust-DE for PINN Hyperparameter Optimization")
    parser.add_argument("benchmark", nargs="?", default="ode", help="Benchmark PDE type (default: ode)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--generations", type=int, default=10, help="Generations count (default: 10)")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (default: 20)")
    parser.add_argument("--steps", type=int, default=1200, help="PINN training steps per candidate (default: 1200)")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced budget (5 gens, pop 6, 150 steps)")

    args = parser.parse_args()

    n_gens = 5 if args.quick else args.generations
    pop = 6 if args.quick else args.pop_size
    n_steps = 150 if args.quick else args.steps

    out_dir = os.path.join("outputs", "pde_robust_de", args.benchmark)
    ensure_dir(out_dir)

    print(f"\n{'='*70}")
    print(f"RUNNING PDE-ROBUST-DE OPTIMIZER ON '{args.benchmark.upper()}' BENCHMARK")
    print(f"Generations: {n_gens} | Pop Size: {pop} | Seed: {args.seed} | PINN Steps: {n_steps}")
    print(f"{'='*70}\n")

    metrics = run_pde_robust_opt(
        out_dir=out_dir,
        benchmark_type=args.benchmark,
        seed=args.seed,
        n_generations=n_gens,
        sol_per_pop=pop,
        n_steps=n_steps,
    )

    print(f"PDE-Robust-DE Results for {args.benchmark.upper()}:")
    print(f"  Final Val Rel L2 = {metrics['val_rel_l2']:.6e}")
    print(f"  Final Val MSE    = {metrics['val_mse']:.6e}")
    print("  Best Config:")
    print(json.dumps(metrics["config"], indent=4))
    print(f"\nDetailed output saved to: {out_dir}/pde_robust_de_best_metrics.json\n")


if __name__ == "__main__":
    main()
