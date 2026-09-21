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
from hpo.novel_f_magso import run_f_magso
from hpo.pde_robust_optimizer import run_pde_robust_opt
from hpo.pso import run_pso
from utils import ensure_dir, save_json


def run_f_magso_ablation(
    benchmark: str, seed: int, quick: bool, n_steps: int, out_dir: str, resume: bool = True
) -> dict[str, dict[str, Any]]:
    pop_size = 6 if quick else 12
    max_evals = 24 if quick else 60
    steps = 60 if quick else n_steps

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
    steps = 60 if quick else n_steps

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
    steps = 60 if quick else n_steps
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
    steps = 60 if quick else n_steps
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
    print(f"\n[+] Full ablation results saved to: {summary_file}\n")


if __name__ == "__main__":
    main()
