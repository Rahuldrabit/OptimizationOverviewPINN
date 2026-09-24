"""Comprehensive Ablation Study Suite for PINN Hyperparameter Optimization.

Ablates and benchmarks:
1. F-MAGSO Components:
   - Full (FLC + GSA + PSO + GA Schema + Multi-Stage)
   - No-FLC (static weights, no fuzzy diversity sensing)
   - No-GSA (no gravitational attraction forces)
   - No-GA-Schema (pure continuous velocity, no discrete architectural recombination)
   - No-MultiStage (static single-phase weights throughout)
2. PDE-Robust-DE Components:
   - Full (JADE adaptive F/CR + bounce-back boundary handling)
   - Fixed-DE (fixed F=0.5, CR=0.7)
   - No-BounceBack (JADE adaptive + standard boundary clipping)
3. Fuzzy-Adaptive Closed-Loop Impact:
   - PSO vs Fuzzy-PSO
   - GA vs Fuzzy-GA
   - ACO vs Fuzzy-ACO
4. Hybrid Synergy Impact:
   - GA-PSO Hybrid vs GA vs PSO
   - PSO-GSA Hybrid vs PSO vs GSA
   - ACO-GA Hybrid vs ACO vs GA

Usage:
    python scripts/run_ablation.py ode --study f_magso --quick
    python scripts/run_ablation.py ode --study pde_robust_de --quick
    python scripts/run_ablation.py ode --study fuzzy --quick
    python scripts/run_ablation.py ode --study hybrids --quick
    python scripts/run_ablation.py ode --study all --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root and src to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from hpo.aco import run_aco
from hpo.fuzzy_aco import run_fuzzy_aco
from hpo.fuzzy_ga import run_fuzzy_ga
from hpo.fuzzy_pso import run_fuzzy_pso
from hpo.ga import run_ga
from hpo.gsa import run_gsa
from hpo.hybrid_aco_ga import run_hybrid_aco_ga
from hpo.hybrid_ga_pso import run_hybrid_ga_pso
from hpo.hybrid_pso_gsa import run_hybrid_pso_gsa
from hpo.f_magso import run_f_magso
from hpo.pde_robust_optimizer import run_pde_robust_opt
from hpo.pso import run_pso
from utils import ensure_dir, save_json


def run_f_magso_ablation(
    benchmark: str, seed: int, quick: bool, n_steps: int, out_dir: str, resume: bool = True
) -> dict[str, dict[str, Any]]:
    pop_size = 6 if quick else 12
    max_evals = 24 if quick else 60
    steps = 1 if quick else n_steps

    variants = {
        "F-MAGSO (Full Proposed)": {
            "use_flc": True,
            "use_gsa": True,
            "use_ga_schema": True,
            "use_multistage": True,
        },
        "F-MAGSO (w/o Fuzzy Controller)": {
            "use_flc": False,
            "use_gsa": True,
            "use_ga_schema": True,
            "use_multistage": True,
        },
        "F-MAGSO (w/o Gravitational GSA)": {
            "use_flc": True,
            "use_gsa": False,
            "use_ga_schema": True,
            "use_multistage": True,
        },
        "F-MAGSO (w/o GA Schema Recombination)": {
            "use_flc": True,
            "use_gsa": True,
            "use_ga_schema": False,
            "use_multistage": True,
        },
        "F-MAGSO (w/o Multi-Stage Transitions)": {
            "use_flc": True,
            "use_gsa": True,
            "use_ga_schema": True,
            "use_multistage": False,
        },
    }

    results: dict[str, dict[str, Any]] = {}
    print(f"\n{'='*70}\n[ABLATION 1] F-MAGSO Architectural Components on '{benchmark.upper()}'\n{'='*70}")

    for name, kwargs in variants.items():
        v_dir = os.path.join(out_dir, "f_magso", name.lower().replace(" ", "_").replace("/", "_"))
        ensure_dir(v_dir)
        ckpt_file = os.path.join(v_dir, "ablation_checkpoint.json")

        if resume and os.path.exists(ckpt_file):
            try:
                with open(ckpt_file, "r", encoding="utf-8") as f:
                    cached_res = json.load(f)
                if cached_res.get("quick") == quick and cached_res.get("n_steps") == steps:
                    results[name] = cached_res
                    print(f"  -> {name} ... (Resumed from checkpoint) | Val Rel L2 = {cached_res['val_rel_l2']:.6e}")
                    continue
            except Exception:
                pass

        print(f"  -> Testing: {name} ...", end="", flush=True)
        t0 = time.perf_counter()
        res = run_f_magso(
            out_dir=v_dir,
            benchmark_type=benchmark,
            seed=seed,
            pop_size=pop_size,
            max_evals=max_evals,
            n_steps=steps,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        res["runtime_sec"] = elapsed
        res["ablation_variant"] = name
        res["quick"] = quick
        res["n_steps"] = steps
        save_json(ckpt_file, res)
        results[name] = res
        print(f" Done in {elapsed:.2f}s | Val Rel L2 = {res['val_rel_l2']:.6e}")

    return results


def run_pde_robust_de_ablation(
    benchmark: str, seed: int, quick: bool, n_steps: int, out_dir: str, resume: bool = True
) -> dict[str, dict[str, Any]]:
    sol_per_pop = 6 if quick else 16
    n_generations = 4 if quick else 8
    steps = 1 if quick else n_steps

    variants = {
        "PDE-Robust-DE (Full Adaptive + BounceBack)": {
            "adaptive": True,
            "bounce_back": True,
        },
        "PDE-Robust-DE (Fixed F=0.5, CR=0.7)": {
            "adaptive": False,
            "bounce_back": True,
        },
        "PDE-Robust-DE (Adaptive + Boundary Clipping)": {
            "adaptive": True,
            "bounce_back": False,
        },
    }

    results: dict[str, dict[str, Any]] = {}
    print(f"\n{'='*70}\n[ABLATION 2] PDE-Robust-DE Adaptation & Boundary Mechanics on '{benchmark.upper()}'\n{'='*70}")

    for name, kwargs in variants.items():
        v_dir = os.path.join(out_dir, "pde_robust_de", name.lower().replace(" ", "_").replace("/", "_"))
        ensure_dir(v_dir)
        ckpt_file = os.path.join(v_dir, "ablation_checkpoint.json")

        if resume and os.path.exists(ckpt_file):
            try:
                with open(ckpt_file, "r", encoding="utf-8") as f:
                    cached_res = json.load(f)
                if cached_res.get("quick") == quick and cached_res.get("n_steps") == steps:
                    results[name] = cached_res
                    print(f"  -> {name} ... (Resumed from checkpoint) | Val Rel L2 = {cached_res['val_rel_l2']:.6e}")
                    continue
            except Exception:
                pass

        print(f"  -> Testing: {name} ...", end="", flush=True)
        t0 = time.perf_counter()
        res = run_pde_robust_opt(
            out_dir=v_dir,
            benchmark_type=benchmark,
            seed=seed,
            n_generations=n_generations,
            sol_per_pop=sol_per_pop,
            n_steps=steps,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        res["runtime_sec"] = elapsed
        res["ablation_variant"] = name
        res["quick"] = quick
        res["n_steps"] = steps
        save_json(ckpt_file, res)
        results[name] = res
        print(f" Done in {elapsed:.2f}s | Val Rel L2 = {res['val_rel_l2']:.6e}")

    return results


def run_fuzzy_ablation(
    benchmark: str, seed: int, quick: bool, n_steps: int, out_dir: str, resume: bool = True
) -> dict[str, dict[str, Any]]:
    steps = 1 if quick else n_steps
    results: dict[str, dict[str, Any]] = {}
    print(f"\n{'='*70}\n[ABLATION 3] Fuzzy Logic Dynamic Adaptation vs Static Baselines on '{benchmark.upper()}'\n{'='*70}")

    pairs = [
        ("PSO (Static)", lambda d: run_pso(d, benchmark, seed=seed, swarmsize=6 if quick else 12, maxiter=3 if quick else 6, n_steps=steps)),
        ("Fuzzy-PSO (Dynamic)", lambda d: run_fuzzy_pso(d, benchmark, seed=seed, swarmsize=6 if quick else 12, maxiter=3 if quick else 6, n_steps=steps)),
        ("GA (Static)", lambda d: run_ga(d, benchmark, seed=seed, n_generations=3 if quick else 6, sol_per_pop=6 if quick else 10, n_steps=steps)),
        ("Fuzzy-GA (Dynamic)", lambda d: run_fuzzy_ga(d, benchmark, seed=seed, n_generations=3 if quick else 6, sol_per_pop=6 if quick else 10, n_steps=steps)),
        ("ACO (Static)", lambda d: run_aco(d, benchmark, seed=seed, n_ants=6 if quick else 10, n_iterations=3 if quick else 6, n_steps=steps)),
        ("Fuzzy-ACO (Dynamic)", lambda d: run_fuzzy_aco(d, benchmark, seed=seed, n_ants=6 if quick else 10, n_iterations=3 if quick else 6, n_steps=steps)),
    ]

    for name, fn in pairs:
        v_dir = os.path.join(out_dir, "fuzzy_ablation", name.lower().replace(" ", "_"))
        ensure_dir(v_dir)
        ckpt_file = os.path.join(v_dir, "ablation_checkpoint.json")

        if resume and os.path.exists(ckpt_file):
            try:
                with open(ckpt_file, "r", encoding="utf-8") as f:
                    cached_res = json.load(f)
                if cached_res.get("quick") == quick and cached_res.get("n_steps") == steps:
                    results[name] = cached_res
                    print(f"  -> {name} ... (Resumed from checkpoint) | Val Rel L2 = {cached_res['val_rel_l2']:.6e}")
                    continue
            except Exception:
                pass

        print(f"  -> Testing: {name} ...", end="", flush=True)
        t0 = time.perf_counter()
        res = fn(v_dir)
        elapsed = time.perf_counter() - t0
        res["runtime_sec"] = elapsed
        res["ablation_variant"] = name
        res["quick"] = quick
        res["n_steps"] = steps
        save_json(ckpt_file, res)
        results[name] = res
        print(f" Done in {elapsed:.2f}s | Val Rel L2 = {res['val_rel_l2']:.6e}")

    return results


def run_hybrids_ablation(
    benchmark: str, seed: int, quick: bool, n_steps: int, out_dir: str, resume: bool = True
) -> dict[str, dict[str, Any]]:
    steps = 1 if quick else n_steps
    results: dict[str, dict[str, Any]] = {}
    print(f"\n{'='*70}\n[ABLATION 4] Hybrid Synergy vs Constituent Standalone Optimizers on '{benchmark.upper()}'\n{'='*70}")

    algs = [
        ("GA (Standalone)", lambda d: run_ga(d, benchmark, seed=seed, n_generations=3 if quick else 6, sol_per_pop=6 if quick else 10, n_steps=steps)),
        ("PSO (Standalone)", lambda d: run_pso(d, benchmark, seed=seed, swarmsize=6 if quick else 12, maxiter=3 if quick else 6, n_steps=steps)),
        ("GA-PSO Hybrid", lambda d: run_hybrid_ga_pso(d, benchmark, seed=seed, pop_size=6 if quick else 10, n_epochs=2 if quick else 3, n_steps=steps)),
        ("GSA (Standalone)", lambda d: run_gsa(d, benchmark, seed=seed, n_agents=6 if quick else 10, n_iterations=3 if quick else 6, n_steps=steps)),
        ("PSO-GSA Hybrid", lambda d: run_hybrid_pso_gsa(d, benchmark, seed=seed, n_agents=6 if quick else 10, n_iterations=3 if quick else 6, n_steps=steps)),
        ("ACO (Standalone)", lambda d: run_aco(d, benchmark, seed=seed, n_ants=6 if quick else 10, n_iterations=3 if quick else 6, n_steps=steps)),
        ("ACO-GA Hybrid", lambda d: run_hybrid_aco_ga(d, benchmark, seed=seed, pop_size=6 if quick else 10, aco_iterations=2 if quick else 3, ga_generations=2 if quick else 3, n_steps=steps)),
    ]

    for name, fn in algs:
        v_dir = os.path.join(out_dir, "hybrid_ablation", name.lower().replace(" ", "_"))
        ensure_dir(v_dir)
        ckpt_file = os.path.join(v_dir, "ablation_checkpoint.json")

        if resume and os.path.exists(ckpt_file):
            try:
                with open(ckpt_file, "r", encoding="utf-8") as f:
                    cached_res = json.load(f)
                if cached_res.get("quick") == quick and cached_res.get("n_steps") == steps:
                    results[name] = cached_res
                    print(f"  -> {name} ... (Resumed from checkpoint) | Val Rel L2 = {cached_res['val_rel_l2']:.6e}")
                    continue
            except Exception:
                pass

        print(f"  -> Testing: {name} ...", end="", flush=True)
        t0 = time.perf_counter()
        res = fn(v_dir)
        elapsed = time.perf_counter() - t0
        res["runtime_sec"] = elapsed
        res["ablation_variant"] = name
        res["quick"] = quick
        res["n_steps"] = steps
        save_json(ckpt_file, res)
        results[name] = res
        print(f" Done in {elapsed:.2f}s | Val Rel L2 = {res['val_rel_l2']:.6e}")

    return results


def print_summary_table(title: str, results: dict[str, dict[str, Any]]) -> None:
    print(f"\n{'-'*75}")
    print(f"SUMMARY: {title}")
    print(f"{'-'*75}")
    print(f"{'Variant / Algorithm':<42} | {'Val Rel L2':<14} | {'Runtime':<8}")
    print(f"{'-'*75}")
    sorted_res = sorted(results.items(), key=lambda item: item[1].get("val_rel_l2", float("inf")))
    for rank, (name, data) in enumerate(sorted_res, start=1):
        err = data.get("val_rel_l2", float("nan"))
        sec = data.get("runtime_sec", 0.0)
        print(f"#{rank} {name:<39} | {err:<14.6e} | {sec:<6.2f}s")
    print(f"{'-'*75}\n")


def generate_ablation_plots(ablation_data: dict[str, Any], output_dir: str) -> dict[str, str]:
    """Generate publication-quality figures for all four ablation studies."""
    ensure_dir(output_dir)
    generated: dict[str, str] = {}

    # 1. F-MAGSO Component Ablation
    f_magso = ablation_data.get("f_magso", {})
    if f_magso:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(f_magso.keys())
        errs = [f_magso[k].get("val_rel_l2", 1.0) for k in names]
        colors = ["#e41a1c" if "Full" in n else "#377eb8" for n in names]
        short_names = [n.replace("F-MAGSO ", "") for n in names]

        bars = ax.barh(short_names, errs, color=colors, alpha=0.85, edgecolor="black", height=0.55)
        ax.set_xscale("log")
        ax.set_xlabel("Validation Relative L2 Error (Log Scale, Lower is Better)", fontsize=11, fontweight="bold")
        ax.set_title("Ablation 1: F-MAGSO Architectural Component Contributions", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", axis="x", linestyle=":", alpha=0.6)

        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.05, bar.get_y() + bar.get_height() / 2, f"{width:.2e}", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        p = os.path.join(output_dir, "f_magso_ablation.png")
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()
        generated["f_magso"] = p

    # 2. PDE-Robust-DE Mechanics Ablation
    pde_de = ablation_data.get("pde_robust_de", {})
    if pde_de:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        names = list(pde_de.keys())
        errs = [pde_de[k].get("val_rel_l2", 1.0) for k in names]
        colors = ["#ff7f00" if "Full" in n else "#4daf4a" for n in names]
        short_names = [n.replace("PDE-Robust-DE ", "") for n in names]

        bars = ax.barh(short_names, errs, color=colors, alpha=0.85, edgecolor="black", height=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Validation Relative L2 Error (Log Scale, Lower is Better)", fontsize=11, fontweight="bold")
        ax.set_title("Ablation 2: PDE-Robust-DE Adaptive Scaling & Boundary Handling", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", axis="x", linestyle=":", alpha=0.6)

        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.05, bar.get_y() + bar.get_height() / 2, f"{width:.2e}", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        p = os.path.join(output_dir, "pde_robust_de_ablation.png")
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()
        generated["pde_robust_de"] = p

    # 3. Fuzzy Dynamic Adaptation vs Static Baselines
    fuzzy = ablation_data.get("fuzzy", {})
    if fuzzy:
        fig, ax = plt.subplots(figsize=(9, 5))
        base_names = ["PSO", "GA", "ACO"]
        static_errs = [fuzzy.get(f"{b} (Static)", {}).get("val_rel_l2", np.nan) for b in base_names]
        dynamic_errs = [fuzzy.get(f"Fuzzy-{b} (Dynamic)", {}).get("val_rel_l2", np.nan) for b in base_names]

        x = np.arange(len(base_names))
        w = 0.35

        ax.bar(x - w / 2, static_errs, w, label="Static Parameter Control", color="#999999", alpha=0.85, edgecolor="black")
        ax.bar(x + w / 2, dynamic_errs, w, label="Fuzzy-Adaptive Closed-Loop", color="#984ea3", alpha=0.85, edgecolor="black")

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(base_names, fontsize=12, fontweight="bold")
        ax.set_ylabel("Validation Relative L2 Error (Log Scale)", fontsize=11, fontweight="bold")
        ax.set_title("Ablation 3: Impact of Mamdani Fuzzy Closed-Loop Dynamic Adaptation", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.6)
        ax.legend(fontsize=11)

        plt.tight_layout()
        p = os.path.join(output_dir, "fuzzy_dynamic_adaptation.png")
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()
        generated["fuzzy"] = p

    # 4. Hybrid Synergy Analysis
    hybrids = ablation_data.get("hybrids", {})
    if hybrids:
        fig, ax = plt.subplots(figsize=(11, 5))
        names = list(hybrids.keys())
        errs = [hybrids[k].get("val_rel_l2", 1.0) for k in names]
        colors = ["#e41a1c" if "Hybrid" in n else "#377eb8" for n in names]

        bars = ax.barh(names, errs, color=colors, alpha=0.85, edgecolor="black", height=0.55)
        ax.set_xscale("log")
        ax.set_xlabel("Validation Relative L2 Error (Log Scale, Lower is Better)", fontsize=11, fontweight="bold")
        ax.set_title("Ablation 4: Hybrid Synergy vs Constituent Standalone Optimizers", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", axis="x", linestyle=":", alpha=0.6)

        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.05, bar.get_y() + bar.get_height() / 2, f"{width:.2e}", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        p = os.path.join(output_dir, "hybrid_synergy_ablation.png")
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()
        generated["hybrids"] = p

    # 5. Master Combined 4-Panel Ablation Summary Figure
    if f_magso and pde_de and fuzzy and hybrids:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Comprehensive Ablation Analysis Across Metaheuristic Components & Mechanics", fontsize=16, fontweight="bold", y=0.99)

        # Panel A: F-MAGSO
        ax_a = axes[0, 0]
        names = list(f_magso.keys())
        errs = [f_magso[k].get("val_rel_l2", 1.0) for k in names]
        colors = ["#e41a1c" if "Full" in n else "#377eb8" for n in names]
        short_names = [n.replace("F-MAGSO ", "") for n in names]
        ax_a.barh(short_names, errs, color=colors, alpha=0.85, edgecolor="black", height=0.55)
        ax_a.set_xscale("log")
        ax_a.set_title("(A) F-MAGSO Component Contributions", fontsize=12, fontweight="bold")
        ax_a.set_xlabel("Validation Rel L2 Error (Log Scale)", fontsize=10)
        ax_a.grid(True, which="both", axis="x", linestyle=":", alpha=0.6)

        # Panel B: PDE-Robust-DE
        ax_b = axes[0, 1]
        names = list(pde_de.keys())
        errs = [pde_de[k].get("val_rel_l2", 1.0) for k in names]
        colors = ["#ff7f00" if "Full" in n else "#4daf4a" for n in names]
        short_names = [n.replace("PDE-Robust-DE ", "") for n in names]
        ax_b.barh(short_names, errs, color=colors, alpha=0.85, edgecolor="black", height=0.5)
        ax_b.set_xscale("log")
        ax_b.set_title("(B) PDE-Robust-DE Adaptation & Boundary", fontsize=12, fontweight="bold")
        ax_b.set_xlabel("Validation Rel L2 Error (Log Scale)", fontsize=10)
        ax_b.grid(True, which="both", axis="x", linestyle=":", alpha=0.6)

        # Panel C: Fuzzy dynamic vs static
        ax_c = axes[1, 0]
        base_names = ["PSO", "GA", "ACO"]
        static_errs = [fuzzy.get(f"{b} (Static)", {}).get("val_rel_l2", np.nan) for b in base_names]
        dynamic_errs = [fuzzy.get(f"Fuzzy-{b} (Dynamic)", {}).get("val_rel_l2", np.nan) for b in base_names]
        x = np.arange(len(base_names))
        w = 0.35
        ax_c.bar(x - w / 2, static_errs, w, label="Static Baseline", color="#999999", alpha=0.85, edgecolor="black")
        ax_c.bar(x + w / 2, dynamic_errs, w, label="Fuzzy-Adaptive", color="#984ea3", alpha=0.85, edgecolor="black")
        ax_c.set_yscale("log")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(base_names, fontsize=11, fontweight="bold")
        ax_c.set_ylabel("Validation Rel L2 Error (Log Scale)", fontsize=10)
        ax_c.set_title("(C) Fuzzy Dynamic Adaptation vs Static Baselines", fontsize=12, fontweight="bold")
        ax_c.grid(True, which="both", axis="y", linestyle=":", alpha=0.6)
        ax_c.legend(fontsize=9)

        # Panel D: Hybrid Synergy
        ax_d = axes[1, 1]
        names = list(hybrids.keys())
        errs = [hybrids[k].get("val_rel_l2", 1.0) for k in names]
        colors = ["#e41a1c" if "Hybrid" in n else "#377eb8" for n in names]
        ax_d.barh(names, errs, color=colors, alpha=0.85, edgecolor="black", height=0.55)
        ax_d.set_xscale("log")
        ax_d.set_title("(D) Hybrid Synergy vs Constituents", fontsize=12, fontweight="bold")
        ax_d.set_xlabel("Validation Rel L2 Error (Log Scale)", fontsize=10)
        ax_d.grid(True, which="both", axis="x", linestyle=":", alpha=0.6)

        plt.tight_layout()
        master_p = os.path.join(output_dir, "comprehensive_ablation_summary.png")
        plt.savefig(master_p, dpi=200, bbox_inches="tight")
        plt.close()
        generated["comprehensive_summary"] = master_p

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PINN HPO Ablation Studies")
    parser.add_argument("benchmark", nargs="?", default="ode", help="Benchmark PDE (ode, heat, burgers, wave)")
    parser.add_argument(
        "--study",
        choices=["f_magso", "pde_robust_de", "fuzzy", "hybrids", "all"],
        default="all",
        help="Ablation study to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--steps", type=int, default=1200, help="PINN training steps (default: 1200)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test mode with reduced budget")
    parser.add_argument("--out-dir", default="outputs/ablation", help="Output directory")

    args = parser.parse_args()
    out_dir = os.path.join(args.out_dir, args.benchmark)
    ensure_dir(out_dir)

    all_data: dict[str, Any] = {}

    if args.study in ["f_magso", "all"]:
        f_magso_res = run_f_magso_ablation(args.benchmark, args.seed, args.quick, args.steps, out_dir)
        print_summary_table("F-MAGSO Component Ablation", f_magso_res)
        all_data["f_magso"] = {k: {"val_rel_l2": v["val_rel_l2"], "runtime_sec": v["runtime_sec"]} for k, v in f_magso_res.items()}

    if args.study in ["pde_robust_de", "all"]:
        pde_de_res = run_pde_robust_de_ablation(args.benchmark, args.seed, args.quick, args.steps, out_dir)
        print_summary_table("PDE-Robust-DE Mechanics Ablation", pde_de_res)
        all_data["pde_robust_de"] = {k: {"val_rel_l2": v["val_rel_l2"], "runtime_sec": v["runtime_sec"]} for k, v in pde_de_res.items()}

    if args.study in ["fuzzy", "all"]:
        fuzzy_res = run_fuzzy_ablation(args.benchmark, args.seed, args.quick, args.steps, out_dir)
        print_summary_table("Fuzzy Dynamic Adaptation Impact", fuzzy_res)
        all_data["fuzzy"] = {k: {"val_rel_l2": v["val_rel_l2"], "runtime_sec": v["runtime_sec"]} for k, v in fuzzy_res.items()}

    if args.study in ["hybrids", "all"]:
        hybrid_res = run_hybrids_ablation(args.benchmark, args.seed, args.quick, args.steps, out_dir)
        print_summary_table("Hybrid Synergy vs Standalones", hybrid_res)
        all_data["hybrids"] = {k: {"val_rel_l2": v["val_rel_l2"], "runtime_sec": v["runtime_sec"]} for k, v in hybrid_res.items()}

    summary_file = os.path.join(out_dir, "ablation_summary.json")
    save_json(summary_file, all_data)
    print(f"\n[+] Full ablation results saved to: {summary_file}")

    # Generate publication figures for ablation studies
    plots_dir = os.path.join(out_dir, "plots")
    print(f"[+] Generating publication ablation figures in: {plots_dir} ...")
    plot_files = generate_ablation_plots(all_data, plots_dir)
    for p_name, p_path in plot_files.items():
        print(f"    - {p_name}: {p_path}")
    print("\n")


if __name__ == "__main__":
    main()
