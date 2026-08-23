"""Master execution script: Runs full comparative investigation of HPO metaheuristics for PINNs.

Benchmarks GA, PSO, ACO, GSA, Fuzzy variants (Fuzzy-PSO, Fuzzy-GA, Fuzzy-ACO),
and Hybrids (GA-PSO, PSO-GSA, ACO-GA). Generates all charts and full Markdown report.
"""

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

from hpo.comparison import ExperimentConfig, run_experiment_grid
from hpo.report_generator import generate_all_plots, generate_markdown_report
from utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full HPO Algorithm Investigation and Generate Report")
    parser.add_argument("--benchmarks", nargs="+", default=["ode", "heat", "burgers", "wave"], help="List of PDE benchmarks to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds for statistical variance")
    parser.add_argument("--steps", type=int, default=1200, help="Training steps per candidate")
    parser.add_argument("--out-dir", default="outputs/comparison", help="Output directory for results & plots")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid smoke test")

    args = parser.parse_args()

    n_steps = 150 if args.quick else args.steps
    config = ExperimentConfig(
        benchmarks=args.benchmarks,
        seeds=args.seeds,
        n_steps=n_steps,
        output_dir=args.out_dir,
    )

    print(f"\n{'='*70}")
    print("PHYSICS-INFORMED NEURAL NETWORK (PINN) HPO INVESTIGATION")
    print(f"{'='*70}\n")

    # 1. Run full experimental grid
    results = run_experiment_grid(config, quick=args.quick, verbose=True)

    # 2. Generate publication-quality plots
    plots_dir = os.path.join(args.out_dir, "plots")
    ensure_dir(plots_dir)
    print(f"\n[+] Generating visualization charts in '{plots_dir}'...")
    plot_files = generate_all_plots(results, plots_dir)
    for name, path in plot_files.items():
        print(f"    - Generated {name.capitalize()} Plot: {path}")

    # 3. Compile full Markdown report
    report_file = os.path.join(args.out_dir, "HPO_INVESTIGATION_REPORT.md")
    print(f"\n[+] Compiling comprehensive report to '{report_file}'...")
    generate_markdown_report(results, plot_files, report_file)
    print(f"    - Complete Report successfully written!")

    print(f"\n{'='*70}")
    print("INVESTIGATION COMPLETE. OVERALL RANKINGS:")
    print(f"{'='*70}")
    for rank, (alg, data) in enumerate(results["overall_rankings"].items(), start=1):
        print(f"  #{rank:2d}: {alg:18s} | Avg Rank: {data['average_rank']:.2f} | Mean Rel L2: {data['overall_mean_rel_l2']:.6f} | Runtime: {data['overall_mean_runtime_sec']:.2f}s")
    print(f"{'='*70}")
    print(f"Full Report: {os.path.abspath(report_file)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
