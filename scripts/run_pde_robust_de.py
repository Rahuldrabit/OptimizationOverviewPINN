"""CLI runner for PDE-Robust-DE (Physics-Informed Differential Evolution with Adaptive Scaling)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean, stdev
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
    parser.add_argument("--seed", type=int, default=None, help="Single random seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Run a reproducible seed batch")
    parser.add_argument("--generations", type=int, default=10, help="Generations count (default: 10)")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (default: 20)")
    parser.add_argument("--steps", type=int, default=1200, help="PINN training steps per candidate (default: 1200)")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced budget (5 gens, pop 6, 150 steps)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: outputs/pde_robust_de/<benchmark>)")

    args = parser.parse_args()

    n_gens = 5 if args.quick else args.generations
    pop = 6 if args.quick else args.pop_size
    n_steps = 150 if args.quick else args.steps

    if args.seeds is not None and args.seed is not None:
        parser.error("Use either --seed or --seeds, not both")
    seeds = args.seeds or [args.seed if args.seed is not None else 0]
    out_dir = args.out_dir or os.path.join("outputs", "pde_robust_de", args.benchmark)
    ensure_dir(out_dir)

    print(f"\n{'='*70}")
    print(f"RUNNING PDE-ROBUST-DE OPTIMIZER ON '{args.benchmark.upper()}' BENCHMARK")
    print(f"Generations: {n_gens} | Pop Size: {pop} | Seeds: {seeds} | PINN Steps: {n_steps}")
    print(f"{'='*70}\n")

    results = []
    for seed in seeds:
        seed_dir = out_dir if len(seeds) == 1 else os.path.join(out_dir, f"seed_{seed}")
        metrics = run_pde_robust_opt(
            out_dir=seed_dir,
            benchmark_type=args.benchmark,
            seed=seed,
            n_generations=n_gens,
            sol_per_pop=pop,
            n_steps=n_steps,
        )
        results.append({"seed": seed, "val_rel_l2": metrics["val_rel_l2"], "val_mse": metrics["val_mse"], "config": metrics["config"]})
        print(f"Seed {seed}: final Val Rel L2 = {metrics['val_rel_l2']:.6e}")

    summary = {
        "benchmark": args.benchmark,
        "seeds": seeds,
        "generations": n_gens,
        "population_size": pop,
        "steps": n_steps,
        "results": results,
        "mean_val_rel_l2": mean(row["val_rel_l2"] for row in results),
        "std_val_rel_l2": stdev(row["val_rel_l2"] for row in results) if len(results) > 1 else 0.0,
    }
    summary_file = os.path.join(out_dir, "pde_robust_de_summary.json")
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSummary saved to: {summary_file}\n")


if __name__ == "__main__":
    main()
