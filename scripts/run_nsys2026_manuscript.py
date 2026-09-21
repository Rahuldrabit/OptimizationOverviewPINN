"""NSYS 2026 manuscript benchmark: GA, PSO, ACO, Fuzzy-GA, Fuzzy-PSO, Fuzzy-ACO
across 4 PDE benchmarks (ODE, Heat, Burgers, Wave) with real trainers.

Records real per-generation convergence AND diversity/exploration trajectories
(see src/hpo/{ga,pso,aco,fuzzy_ga,fuzzy_pso,fuzzy_aco}.py) for every algorithm,
not just the fuzzy-adaptive ones. Runs are dispatched across a process pool since
each (benchmark, algorithm, seed) combination is independent - on a 12-core
machine this turns a ~24h serial run into roughly 3 hours wall-clock.

Usage:
    python scripts/run_nsys2026_manuscript.py                  # full: 3 seeds, 1200 steps
    python scripts/run_nsys2026_manuscript.py --quick           # smoke test: small budgets
    python scripts/run_nsys2026_manuscript.py --workers 4       # override parallelism
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Smoke-test budgets (small pop/gens, short training)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds (default: project standard 3 seeds)")
    parser.add_argument("--n-steps", type=int, default=1200, help="PINN training steps per candidate evaluation")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)),
                         help="Parallel worker processes (default: cpu_count-2, capped at 8)")
    parser.add_argument("--output-dir", default="outputs/nsys2026", help="Output directory for results/plots/report")
    args = parser.parse_args()

    from hpo.comparison import ExperimentConfig, run_experiment_grid
    from hpo.report_generator import generate_all_plots, generate_markdown_report
    from utils import ensure_dir

    config = ExperimentConfig(
        benchmarks=["ode", "heat", "burgers", "wave"],  # the only benchmarks with real (non-placeholder) trainers
        algorithms=["GA", "PSO", "ACO", "Fuzzy-GA", "Fuzzy-PSO", "Fuzzy-ACO"],
        seeds=args.seeds,
        n_steps=1 if args.quick else args.n_steps,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("NSYS 2026 MANUSCRIPT BENCHMARK: GA / PSO / ACO / Fuzzy-GA / Fuzzy-PSO / Fuzzy-ACO")
    print(f"Benchmarks : {config.benchmarks}")
    print(f"Seeds      : {config.seeds}")
    print(f"n_steps    : {config.n_steps}")
    print(f"Workers    : {args.workers}")
    print("=" * 70 + "\n")

    results = run_experiment_grid(config, quick=args.quick, verbose=True, max_workers=args.workers)

    plots_dir = os.path.join(config.output_dir, "plots")
    ensure_dir(plots_dir)
    print(f"\n[+] Generating plots in '{plots_dir}'...")
    plot_files = generate_all_plots(results, plots_dir)
    for name, path in plot_files.items():
        print(f"    - {name.capitalize()}: {path}")

    report_file = os.path.join(config.output_dir, "MANUSCRIPT_REPORT.md")
    print(f"\n[+] Writing report to '{report_file}'...")
    generate_markdown_report(results, plot_files, report_file)

    print("\n" + "=" * 70)
    print("NSYS 2026 MANUSCRIPT RESULTS READY")
    print("=" * 70)
    for rank, (alg, data) in enumerate(results["overall_rankings"].items(), start=1):
        div = data.get("overall_mean_diversity", float("nan"))
        print(f"  #{rank}: {alg:12s} Avg Rank: {data['average_rank']:.2f} | Mean L2: {data['overall_mean_rel_l2']:.6f} | Mean Diversity: {div:.3f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
