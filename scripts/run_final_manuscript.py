"""Unified Master Execution Suite for PINN Hyperparameter Optimization Manuscript.

Executes the complete experimental pipeline in one command:
1. Multi-PDE Benchmark Grid (Baselines, Fuzzy variants, Hybrids, F-MAGSO, and PDE-Robust-DE).
2. Systematic Ablation Studies (F-MAGSO components, PDE-Robust-DE mechanics, Fuzzy adaptations, Hybrid synergies).
3. Publication Figure Generation (Convergence trajectories, radar charts, rankings heatmap, boxplots).
4. Automated LaTeX Tables Export (Rankings, Head-to-Head, Speed, and Ablation tables).
5. Comprehensive Final Markdown Report.

Usage:
    python scripts/run_final_manuscript.py                  # Full production run (3 seeds, 1200 steps)
    python scripts/run_final_manuscript.py --quick          # Smoke test across all components
    python scripts/run_final_manuscript.py --workers 4      # Custom worker parallelism
    python scripts/run_final_manuscript.py --skip-ablation  # Run benchmark grid only
    python scripts/run_final_manuscript.py --skip-grid      # Run ablation studies only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root and src are on sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from hpo.comparison import ExperimentConfig, run_experiment_grid
from hpo.report_generator import generate_all_plots, generate_markdown_report
from utils import ensure_dir, save_json

from scripts.run_ablation import (
    run_f_magso_ablation,
    run_pde_robust_de_ablation,
    run_fuzzy_ablation,
    run_hybrids_ablation,
    generate_ablation_plots,
    print_summary_table,
)
from hpo.speed_benchmark import run_full_convergence_speed_benchmark
from scripts.export_latex_tables import (
    export_ranking_latex_table,
    export_speed_latex_table,
    export_head_to_head_latex_table,
    sync_figures_to_paper,
)


def export_ablation_latex_table(ablation_data: dict[str, Any], out_file: str) -> None:
    """Generate publication-quality LaTeX table summarizing all ablation studies."""
    ensure_dir(os.path.dirname(out_file))

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation Study Results: Performance degradation and runtime across architectural components, mechanics, fuzzy adaptation, and hybrid formulations.}",
        r"\label{tab:ablation_results}",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Study Group} & \textbf{Variant Description} & \textbf{Relative $L_2$ Error} & \textbf{Runtime (s)} \\",
        r"\midrule",
    ]

    group_labels = {
        "f_magso": "F-MAGSO Component Ablation",
        "pde_robust_de": "PDE-Robust-DE Mechanics",
        "fuzzy": "Fuzzy Closed-Loop Adaptation",
        "hybrids": "Hybrid Synergy Analysis",
    }

    for group_key, group_title in group_labels.items():
        items = ablation_data.get(group_key, {})
        if not items:
            continue

        latex_lines.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{group_title}}}}} \\")
        latex_lines.append(r"\midrule")

        sorted_items = sorted(items.items(), key=lambda x: x[1].get("val_rel_l2", float("inf")))
        for name, metrics in sorted_items:
            err = metrics.get("val_rel_l2", float("nan"))
            sec = metrics.get("runtime_sec", 0.0)

            # Highlight proposed/full variants
            if "Full" in name or "(Dynamic)" in name or "Hybrid" in name:
                name_fmt = rf"\textbf{{{name}}}"
            else:
                name_fmt = name

            latex_lines.append(
                f" & {name_fmt:<45s} & {err:12.6e} & {sec:6.2f}s \\\\"
            )
        latex_lines.append(r"\midrule")

    # Remove trailing midrule if any and close table
    if latex_lines[-1] == r"\midrule":
        latex_lines.pop()

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"[+] Exported LaTeX Ablation Table: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master Execution Script: Run all PINN HPO Benchmarks, Ablation Studies, and Generate Paper Artifacts."
    )
    parser.add_argument("--quick", action="store_true", help="Quick smoke-test mode with reduced budgets and iterations")
    parser.add_argument("--full", action="store_true", help="Explicitly request full manuscript scale (3 seeds, 1200 steps)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds (default: [0, 1, 2] for full, [0] for quick)")
    parser.add_argument("--steps", type=int, default=1200, help="PINN training steps per candidate evaluation (default: 1200)")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["ode", "heat", "burgers", "wave"],
        help="PDE benchmarks to evaluate (ode, heat, burgers, wave)",
    )
    parser.add_argument(
        "--ablation-benchmarks",
        nargs="+",
        default=["ode"],
        help="PDE benchmarks to run ablation studies on (default: ode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4) - 2)),
        help="Parallel worker processes for benchmark grid (default: cpu_count-2, capped at 8)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript",
        help="Base directory for all manuscript results, plots, and tables",
    )
    parser.add_argument("--skip-grid", action="store_true", help="Skip the main benchmark grid")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip the ablation studies")
    parser.add_argument("--skip-speed", action="store_true", help="Skip the convergence speed benchmark")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoints; force re-running all tasks from scratch",
    )
    args = parser.parse_args()

    # Determine seeds and steps based on mode
    if args.quick:
        seeds = args.seeds or [0]
        n_steps = 150
    else:
        seeds = args.seeds or [0, 1, 2]
        n_steps = args.steps

    resume = not args.no_resume
    base_out = args.output_dir
    ensure_dir(base_out)
    latex_dir = str(project_root / "paper" / "tables")
    ensure_dir(latex_dir)

    print("\n" + "=" * 80)
    print("      FINAL MANUSCRIPT MASTER EXPERIMENTAL SUITE")
    print("=" * 80)
    print(f"Mode                : {'QUICK SMOKE TEST' if args.quick else 'FULL PRODUCTION BENCHMARK'}")
    print(f"PDE Benchmarks      : {args.benchmarks}")
    print(f"Ablation Benchmarks : {args.ablation_benchmarks}")
    print(f"Random Seeds        : {seeds}")
    print(f"PINN Steps          : {n_steps}")
    print(f"Workers (Parallel)  : {args.workers}")
    print(f"Resume Checkpoints  : {resume}")
    print(f"Output Directory    : {base_out}")
    print("=" * 80 + "\n")

    overall_start_time = time.perf_counter()

    # =========================================================================
    # PART 1: Comprehensive Multi-Algorithm Benchmark Grid
    # =========================================================================
    grid_results = None
    if not args.skip_grid:
        print("\n" + "#" * 80)
        print("STAGE 1: COMPREHENSIVE BENCHMARK GRID (13 OPTIMIZATION ALGORITHMS)")
        print("#" * 80 + "\n")

        all_algorithms = [
            "GA", "PSO", "ACO", "GSA",
            "Fuzzy-GA", "Fuzzy-PSO", "Fuzzy-ACO",
            "GA-PSO Hybrid", "PSO-GSA Hybrid", "ACO-GA Hybrid",
            "F-MAGSO",
            "PDE-Robust-DE",
            "Two-Stage Evo (Buzaev 2026)",
        ]

        grid_config = ExperimentConfig(
            benchmarks=args.benchmarks,
            algorithms=all_algorithms,
            seeds=seeds,
            n_steps=1 if args.quick else n_steps,
            output_dir=base_out,
        )

        grid_results = run_experiment_grid(
            grid_config,
            quick=args.quick,
            verbose=True,
            max_workers=args.workers,
            resume=resume,
        )

        # Generate publication figures
        plots_dir = os.path.join(base_out, "plots")
        ensure_dir(plots_dir)
        print(f"\n[+] Generating manuscript charts in '{plots_dir}'...")
        plot_files = generate_all_plots(grid_results, plots_dir)
        for name, path in plot_files.items():
            print(f"    - {name.capitalize()}: {path}")

        # Compile final markdown report
        report_file = os.path.join(base_out, "FINAL_MANUSCRIPT_REPORT.md")
        print(f"\n[+] Writing comprehensive report to '{report_file}'...")
        generate_markdown_report(grid_results, plot_files, report_file)

        # Export ranking table
        export_ranking_latex_table(grid_results, os.path.join(latex_dir, "table_ranking.tex"))
    else:
        print("\n[!] Skipping Stage 1 (Benchmark Grid) as requested.")

    # =========================================================================
    # PART 2: Systematic Ablation Studies
    # =========================================================================
    ablation_results: dict[str, Any] = {}
    if not args.skip_ablation:
        print("\n" + "#" * 80)
        print("STAGE 2: SYSTEMATIC ABLATION STUDIES")
        print("#" * 80 + "\n")

        ablation_dir = os.path.join(base_out, "ablation")
        ensure_dir(ablation_dir)
        primary_ablation_bmark = args.ablation_benchmarks[0]
        ablation_seed = seeds[0]

        # 1. F-MAGSO Component Ablation
        print("\n[1/4] Running F-MAGSO Component Ablation...")
        f_magso_res = run_f_magso_ablation(
            primary_ablation_bmark, ablation_seed, args.quick, n_steps, ablation_dir, resume=resume
        )
        print_summary_table("F-MAGSO Architectural Components", f_magso_res)
        ablation_results["f_magso"] = {
            k: {"val_rel_l2": v["val_rel_l2"], "runtime_sec": v["runtime_sec"]}
            for k, v in f_magso_res.items()
        }

        # 2. PDE-Robust-DE Mechanics Ablation
        print("\n[2/4] Running PDE-Robust-DE Mechanics Ablation...")
        pde_de_res = run_pde_robust_de_ablation(
            primary_ablation_bmark, ablation_seed, args.quick, n_steps, ablation_dir, resume=resume
        )
        print_summary_table("PDE-Robust-DE Mechanics", pde_de_res)
        ablation_results["pde_robust_de"] = {
            k: {"val_rel_l2": v["val_rel_l2"], "runtime_sec": v["runtime_sec"]}
            for k, v in pde_de_res.items()
        }

        # 3. Fuzzy Closed-Loop Dynamic Adaptation Ablation
        print("\n[3/4] Running Fuzzy Dynamic Adaptation vs Static Baselines...")
        fuzzy_res = run_fuzzy_ablation(
            primary_ablation_bmark, ablation_seed, args.quick, n_steps, ablation_dir, resume=resume
        )
        print_summary_table("Fuzzy Closed-Loop Adaptation Impact", fuzzy_res)
        ablation_results["fuzzy"] = {
            k: {"val_rel_l2": v["val_rel_l2"], "runtime_sec": v["runtime_sec"]}
            for k, v in fuzzy_res.items()
        }

        # 4. Hybrid Synergy Ablation
        print("\n[4/4] Running Hybrid Synergy Analysis vs Constituent Standalones...")
        hybrid_res = run_hybrids_ablation(
            primary_ablation_bmark, ablation_seed, args.quick, n_steps, ablation_dir, resume=resume
        )
        print_summary_table("Hybrid Synergy vs Constituent Optimizers", hybrid_res)
        ablation_results["hybrids"] = {
            k: {"val_rel_l2": v["val_rel_l2"], "runtime_sec": v["runtime_sec"]}
            for k, v in hybrid_res.items()
        }

        # Save summary JSON
        ablation_summary_file = os.path.join(ablation_dir, "ablation_summary.json")
        save_json(ablation_summary_file, ablation_results)
        print(f"\n[+] Ablation summary saved to: {ablation_summary_file}")

        # Generate publication figures for ablation studies
        plots_dir = os.path.join(base_out, "plots")
        ensure_dir(plots_dir)
        print(f"\n[+] Generating publication ablation figures in '{plots_dir}'...")
        generate_ablation_plots(ablation_results, plots_dir)
        generate_ablation_plots(ablation_results, os.path.join(ablation_dir, "plots"))

        # Export ablation LaTeX table
        export_ablation_latex_table(ablation_results, os.path.join(latex_dir, "table_ablation.tex"))
    else:
        print("\n[!] Skipping Stage 2 (Ablation Studies) as requested.")

    # =========================================================================
    # PART 2.5: High-Resolution Convergence Speed Benchmark
    # =========================================================================
    if not args.skip_speed:
        print("\n" + "#" * 80)
        print("STAGE 2.5: HIGH-RESOLUTION CONVERGENCE SPEED BENCHMARK (13 ALGORITHMS)")
        print("#" * 80 + "\n")
        speed_dir = os.path.join(base_out, "speed_benchmark")
        ensure_dir(speed_dir)
        speed_seeds = seeds if len(seeds) > 1 else [0]
        speed_evals = 30 if args.quick else 60
        run_full_convergence_speed_benchmark(
            benchmark_type=args.benchmarks[0],
            seeds=speed_seeds,
            max_evals=speed_evals,
            output_dir=speed_dir,
            n_steps=1 if args.quick else n_steps,
        )
    else:
        print("\n[!] Skipping Stage 2.5 (Convergence Speed Benchmark) as requested.")

    # =========================================================================
    # PART 3: Publication Artifacts Synchronization
    # =========================================================================
    print("\n" + "#" * 80)
    print("STAGE 3: EXPORTING PUBLICATION LATEX TABLES & FIGURES")
    print("#" * 80 + "\n")

    # Export speed and head-to-head baseline tables
    export_speed_latex_table(os.path.join(latex_dir, "table_speed.tex"))
    export_head_to_head_latex_table(os.path.join(latex_dir, "table_baseline_comparison.tex"))

    # Sync plots from base_out/plots to paper/figures
    sync_figures_to_paper()

    total_time = time.perf_counter() - overall_start_time
    print("\n" + "=" * 80)
    print("                 ALL MANUSCRIPT WORKFLOWS COMPLETE!")
    print("=" * 80)
    print(f"Total Execution Time : {total_time / 60:.2f} minutes ({total_time:.1f}s)")
    print(f"Results Directory    : {os.path.abspath(base_out)}")
    print(f"LaTeX Tables         : {os.path.abspath(latex_dir)}")
    print(f"Paper Figures        : {os.path.abspath(project_root / 'paper' / 'figures')}")
    if grid_results and "overall_rankings" in grid_results:
        print("\nTOP PERFORMING ALGORITHMS (FRIEDMAN RANK):")
        for rank, (alg, data) in enumerate(grid_results["overall_rankings"].items(), start=1):
            if rank <= 5:
                print(f"  #{rank}: {alg:26s} Avg Rank: {data['average_rank']:.2f} | Mean Rel L2: {data['overall_mean_rel_l2']:.6e}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
