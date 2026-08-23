"""One-command reproducibility script for the full PINN HPO paper results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent


def run_command(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*70}\n[STEP] {desc}\nCMD: {' '.join(cmd)}\n{'='*70}")
    res = subprocess.run(cmd, cwd=str(project_root))
    if res.returncode != 0:
        print(f"[!] Error in step: {desc} (Exit code {res.returncode})")
        sys.exit(res.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce all empirical benchmarks, plots, and LaTeX tables for the PINN HPO paper.")
    parser.add_argument("--quick", action="store_true", help="Run in quick smoke-test mode with smaller budgets")
    args = parser.parse_args()

    python_exe = sys.executable

    # 1. Multi-PDE Comparison Grid
    quick_flag = ["--quick"] if args.quick else []
    run_command(
        [python_exe, "scripts/run_full_comparison.py"] + quick_flag,
        "Running 14-Algorithm Multi-PDE Benchmark Grid (ODE, Heat, Burgers, Wave)"
    )

    # 2. Convergence Speed Benchmark
    speed_evals = ["--evals", "30"] if args.quick else ["--evals", "60"]
    run_command(
        [python_exe, "scripts/test_convergence_speed.py"] + speed_evals,
        "Running High-Resolution Convergence Speed & Trajectory Benchmark"
    )

    # 3. Export LaTeX Tables
    run_command(
        [python_exe, "scripts/export_latex_tables.py"],
        "Exporting Publication-Ready LaTeX Tables"
    )

    print(f"\n{'='*70}\n[SUCCESS] ALL BENCHMARKS, PLOTS, AND LATEX TABLES REPRODUCED SUCCESSFULLY!\n{'='*70}\n")


if __name__ == "__main__":
    main()
