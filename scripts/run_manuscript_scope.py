"""Focused manuscript scope: 7 core algorithms × 4 benchmarks × 2 seeds = 56 runs
Covers: 3 proposed methods, SOTA baseline, 3 standalone comparisons.
Estimated: 12-24 hours of continuous CPU time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from hpo.comparison import ExperimentConfig, run_experiment_grid
from hpo.report_generator import generate_all_plots, generate_markdown_report
from utils import ensure_dir


def main() -> None:
    config = ExperimentConfig(
        benchmarks=["ode", "heat", "burgers", "wave"],
        algorithms=[
            "GA", "PSO", "ACO",  # Standalones
            "Fuzzy-PSO",  # Proposed
            "ACO-GA Hybrid",  # Proposed
            "PDE-Robust-DE",  # Proposed
            "Two-Stage Evo (Buzaev 2026)",  # SOTA baseline
        ],
        seeds=[0, 1],  # 2 seeds for statistical validity
        n_steps=1200,
        output_dir="outputs/manuscript_scope",
    )

    print("\n" + "="*70)
    print("MANUSCRIPT-FOCUSED HPO BENCHMARK (7 algorithms × 4 benchmarks × 2 seeds)")
    print("="*70 + "\n")

    results = run_experiment_grid(config, quick=False, verbose=True)

    # Generate plots
    plots_dir = os.path.join(config.output_dir, "plots")
    ensure_dir(plots_dir)
    print(f"\n[+] Generating plots in '{plots_dir}'...")
    plot_files = generate_all_plots(results, plots_dir)
    for name, path in plot_files.items():
        print(f"    ✓ {name.capitalize()}: {path}")

    # Generate report
    report_file = os.path.join(config.output_dir, "MANUSCRIPT_REPORT.md")
    print(f"\n[+] Writing report to '{report_file}'...")
    generate_markdown_report(results, plot_files, report_file)

    print("\n" + "="*70)
    print("MANUSCRIPT RESULTS READY")
    print("="*70)
    for rank, (alg, data) in enumerate(results["overall_rankings"].items(), start=1):
        print(f"  #{rank}: {alg:25s} Avg Rank: {data['average_rank']:.2f} | Mean L2: {data['overall_mean_rel_l2']:.6f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
