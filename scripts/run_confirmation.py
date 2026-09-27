"""Run a controlled confirmation study with equal candidate-evaluation budgets.

Example:
    python scripts/run_confirmation.py --steps 1200 --evals 80
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from hpo.speed_benchmark import run_full_convergence_speed_benchmark
from utils import ensure_dir, save_json

DEFAULT_ALGORITHMS = [
    "PDE-Robust-DE", "GA", "PSO", "ACO", "GSA", "Two-Stage Evo (Buzaev 2026)"
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run equal-budget confirmation experiments")
    parser.add_argument("--benchmarks", nargs="+", default=["ode", "heat", "burgers", "wave"])
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--evals", type=int, default=80, help="Exact candidate trainings per run")
    parser.add_argument("--steps", type=int, default=1200, help="PINN training steps per candidate")
    parser.add_argument("--out-dir", default="outputs/confirmation_10seeds")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    combined: dict[str, Any] = {
        "metadata": {
            "benchmarks": args.benchmarks,
            "algorithms": args.algorithms,
            "seeds": args.seeds,
            "max_evals": args.evals,
            "n_steps": args.steps,
        },
        "raw_runs": {},
        "aggregated": {},
    }

    for benchmark in args.benchmarks:
        benchmark_dir = os.path.join(args.out_dir, benchmark)
        print(f"\nRunning {benchmark.upper()}: {len(args.algorithms)} algorithms, {len(args.seeds)} seeds, {args.evals} evaluations/run")
        result = run_full_convergence_speed_benchmark(
            benchmark_type=benchmark,
            seeds=args.seeds,
            max_evals=args.evals,
            output_dir=benchmark_dir,
            n_steps=args.steps,
            algorithms=args.algorithms,
        )
        combined["raw_runs"][benchmark] = result["raw_runs"]
        combined["aggregated"][benchmark] = result["aggregated"]

    output_file = os.path.join(args.out_dir, "confirmation_results.json")
    save_json(output_file, combined)
    print(f"\nSaved controlled confirmation results to: {output_file}")
    print("Every algorithm/benchmark/seed cell used the same max evaluation budget.")


if __name__ == "__main__":
    main()
