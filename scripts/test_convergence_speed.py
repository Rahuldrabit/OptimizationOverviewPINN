"""CLI script to run high-resolution convergence speed tests and generate speed comparison charts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from hpo.speed_benchmark import run_full_convergence_speed_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and compare convergence speed across all HPO algorithms")
    parser.add_argument("benchmark", nargs="?", default="ode", help="Benchmark PDE type (default: ode)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Random seeds for testing")
    parser.add_argument("--evals", type=int, default=60, help="Maximum function evaluations per run")
    parser.add_argument("--out-dir", default="outputs/speed_benchmark", help="Output directory for speed report & plots")

    args = parser.parse_args()

    print(f"\n{'='*75}")
    print(f"TESTING OPTIMIZATION CONVERGENCE SPEED ON '{args.benchmark.upper()}'")
    print(f"Seeds: {args.seeds} | Max Evals/Run: {args.evals}")
    print(f"{'='*75}\n")

    results = run_full_convergence_speed_benchmark(
        benchmark_type=args.benchmark,
        seeds=args.seeds,
        max_evals=args.evals,
        output_dir=args.out_dir,
    )

    print(f"\n{'='*75}")
    print("CONVERGENCE SPEED RANKING SUMMARY (FASTEST TO SLOWEST):")
    print(f"{'='*75}")
    for rank, alg in enumerate(results["speed_ranking"], start=1):
        data = results["aggregated"][alg]
        ev_02 = f"{data['avg_evals_to_0.02']:.1f}" if data['avg_evals_to_0.02'] < float("inf") else "> max"
        print(f"  #{rank:2d}: {alg:18s} | Evals to Error < 0.02: {ev_02:7s} | Initial Descent: {data['mean_descent_velocity']:.4f}/eval | AUC: {data['mean_auc']:.2f}")
    print(f"{'='*75}")
    print(f"Full Report: {os.path.abspath(os.path.join(args.out_dir, 'CONVERGENCE_SPEED_REPORT.md'))}")
    print(f"Plots: {os.path.abspath(os.path.join(args.out_dir, 'plots'))}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
