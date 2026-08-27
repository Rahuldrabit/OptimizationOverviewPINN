"""Unified Comparison Engine for HPO Metaheuristic Benchmarking.

Runs config-style experiments across multiple algorithms, seeds, and PDE benchmarks,
recording convergence metrics, runtime, and discovered hyperparameters.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

try:
    from ..utils import ensure_dir, save_json
    from .ga import run_ga
    from .pso import run_pso
    from .aco import run_aco
    from .gsa import run_gsa
    from .fuzzy_ga import run_fuzzy_ga
    from .fuzzy_pso import run_fuzzy_pso
    from .fuzzy_aco import run_fuzzy_aco
    from .hybrid_ga_pso import run_hybrid_ga_pso
    from .hybrid_pso_gsa import run_hybrid_pso_gsa
    from .hybrid_aco_ga import run_hybrid_aco_ga
    from .novel_f_magso import run_f_magso
    from .omnipinn_optimizer import run_omnipinn_opt
    from .pde_robust_optimizer import run_pde_robust_opt
    from .two_stage_evo import run_two_stage_evo
except (ImportError, ValueError):
    from utils import ensure_dir, save_json
    from hpo.ga import run_ga
    from hpo.pso import run_pso
    from hpo.aco import run_aco
    from hpo.gsa import run_gsa
    from hpo.fuzzy_ga import run_fuzzy_ga
    from hpo.fuzzy_pso import run_fuzzy_pso
    from hpo.fuzzy_aco import run_fuzzy_aco
    from hpo.hybrid_ga_pso import run_hybrid_ga_pso
    from hpo.hybrid_pso_gsa import run_hybrid_pso_gsa
    from hpo.hybrid_aco_ga import run_hybrid_aco_ga
    from hpo.novel_f_magso import run_f_magso
    from hpo.omnipinn_optimizer import run_omnipinn_opt
    from hpo.pde_robust_optimizer import run_pde_robust_opt
    from hpo.two_stage_evo import run_two_stage_evo



@dataclass
class ExperimentConfig:
    """Config-style search configuration."""
    benchmarks: list[str] = field(default_factory=lambda: ["ode", "heat", "burgers", "wave"])
    algorithms: list[str] = field(default_factory=lambda: [
        "GA", "PSO", "ACO", "GSA",
        "Fuzzy-GA", "Fuzzy-PSO", "Fuzzy-ACO",
        "GA-PSO Hybrid", "PSO-GSA Hybrid", "ACO-GA Hybrid",
        "F-MAGSO (Novel)",
        "PDE-Robust-DE",
        "Two-Stage Evo (Buzaev 2026)"
    ])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_steps: int = 1200
    n_eval_budget: int = 100  # approximate candidate evaluation budget per run
    output_dir: str = "outputs/comparison"


ALGORITHM_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "GA": lambda out_dir, bmark, seed, steps, quick: run_ga(
        out_dir, bmark, seed=seed,
        n_generations=4 if quick else 8,
        sol_per_pop=6 if quick else 10,
        n_steps=steps
    ),
    "PSO": lambda out_dir, bmark, seed, steps, quick: run_pso(
        out_dir, bmark, seed=seed,
        swarmsize=6 if quick else 12,
        maxiter=4 if quick else 8,
        n_steps=steps
    ),
    "ACO": lambda out_dir, bmark, seed, steps, quick: run_aco(
        out_dir, bmark, seed=seed,
        n_ants=6 if quick else 10,
        n_iterations=4 if quick else 8,
        n_steps=steps
    ),
    "GSA": lambda out_dir, bmark, seed, steps, quick: run_gsa(
        out_dir, bmark, seed=seed,
        n_agents=6 if quick else 12,
        n_iterations=4 if quick else 8,
        n_steps=steps
    ),
    "Fuzzy-GA": lambda out_dir, bmark, seed, steps, quick: run_fuzzy_ga(
        out_dir, bmark, seed=seed,
        n_generations=4 if quick else 8,
        sol_per_pop=6 if quick else 10,
        n_steps=steps
    ),
    "Fuzzy-PSO": lambda out_dir, bmark, seed, steps, quick: run_fuzzy_pso(
        out_dir, bmark, seed=seed,
        swarmsize=6 if quick else 12,
        maxiter=4 if quick else 8,
        n_steps=steps
    ),
    "Fuzzy-ACO": lambda out_dir, bmark, seed, steps, quick: run_fuzzy_aco(
        out_dir, bmark, seed=seed,
        n_ants=6 if quick else 10,
        n_iterations=4 if quick else 8,
        n_steps=steps
    ),
    "GA-PSO Hybrid": lambda out_dir, bmark, seed, steps, quick: run_hybrid_ga_pso(
        out_dir, bmark, seed=seed,
        pop_size=6 if quick else 12,
        n_epochs=2 if quick else 4,
        ga_gens_per_epoch=1 if quick else 2,
        pso_iters_per_epoch=1 if quick else 2,
        n_steps=steps
    ),
    "PSO-GSA Hybrid": lambda out_dir, bmark, seed, steps, quick: run_hybrid_pso_gsa(
        out_dir, bmark, seed=seed,
        n_agents=6 if quick else 12,
        n_iterations=4 if quick else 8,
        n_steps=steps
    ),
    "ACO-GA Hybrid": lambda out_dir, bmark, seed, steps, quick: run_hybrid_aco_ga(
        out_dir, bmark, seed=seed,
        pop_size=6 if quick else 12,
        aco_iterations=2 if quick else 4,
        ga_generations=2 if quick else 4,
        n_steps=steps
    ),
    "F-MAGSO (Novel)": lambda out_dir, bmark, seed, steps, quick: run_f_magso(
        out_dir, bmark, seed=seed,
        pop_size=6 if quick else 12,
        max_evals=30 if quick else 80,
        n_steps=steps
    ),
    "PDE-Robust-DE": lambda out_dir, bmark, seed, steps, quick: run_pde_robust_opt(
        out_dir, bmark, seed=seed,
        n_generations=5 if quick else 10,
        sol_per_pop=6 if quick else 20,
        n_steps=steps
    ),
    "Two-Stage Evo (Buzaev 2026)": lambda out_dir, bmark, seed, steps, quick: run_two_stage_evo(
        out_dir, bmark, seed=seed,
        max_evals=30 if quick else 80,
        n_steps=steps,
    ),
}



def _run_single_job(alg: str, out_dir: str, bmark: str, seed: int, n_steps: int, quick: bool) -> tuple[dict[str, Any], float]:
    """Module-level worker: runs one (benchmark, algorithm, seed) job and times it.

    Kept at module scope (rather than inline) so it can be pickled by reference when
    dispatched through ProcessPoolExecutor with the "spawn" start method (required on
    Windows) - only the algorithm name and plain arguments cross the process boundary,
    not the ALGORITHM_REGISTRY lambda itself.
    """
    try:
        import torch
        torch.set_num_threads(1)  # avoid N-worker x M-thread oversubscription across cores
    except ImportError:
        pass
    ensure_dir(out_dir)
    runner_fn = ALGORITHM_REGISTRY[alg]
    t0 = time.perf_counter()
    run_metrics = runner_fn(out_dir, bmark, seed, n_steps, quick)
    elapsed = time.perf_counter() - t0
    return run_metrics, elapsed


def run_experiment_grid(
    config: ExperimentConfig,
    quick: bool = False,
    verbose: bool = True,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Execute the full config-style search across algorithms, benchmarks, and seeds.

    max_workers > 1 dispatches independent (benchmark, algorithm, seed) jobs across a
    ProcessPoolExecutor to use multiple CPU cores; max_workers=1 preserves the original
    strictly-serial behavior.
    """
    ensure_dir(config.output_dir)
    results: dict[str, Any] = {
        "metadata": {
            "benchmarks": config.benchmarks,
            "algorithms": config.algorithms,
            "seeds": config.seeds,
            "quick": quick,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "raw_runs": {},
        "benchmark_summary": {},
        "overall_rankings": {},
    }

    total_runs = len(config.benchmarks) * len(config.algorithms) * len(config.seeds)
    current_run = 0

    if verbose:
        print("=" * 70)
        print(f"STARTING COMPREHENSIVE HPO BENCHMARK SUITE ({total_runs} total runs)")
        print(f"Benchmarks: {config.benchmarks}")
        print(f"Algorithms: {config.algorithms}")
        print(f"Seeds: {config.seeds}")
        print(f"Parallel workers: {max_workers}")
        print("=" * 70)

    # Job specs: (bmark, alg, seed, out_dir), skipping unknown algorithms up front.
    job_specs: list[tuple[str, str, int, str]] = []
    for bmark in config.benchmarks:
        for alg in config.algorithms:
            if alg not in ALGORITHM_REGISTRY:
                print(f"Warning: algorithm '{alg}' not found in registry, skipping.")
                continue
            for seed in config.seeds:
                out_dir = os.path.join(config.output_dir, "runs", bmark, alg.lower().replace(" ", "_"), f"seed_{seed}")
                job_specs.append((bmark, alg, seed, out_dir))

    job_results: dict[tuple[str, str, int], tuple[dict[str, Any], float]] = {}

    if max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_single_job, alg, out_dir, bmark, seed, config.n_steps, quick): (bmark, alg, seed)
                for (bmark, alg, seed, out_dir) in job_specs
            }
            for future in as_completed(futures):
                bmark, alg, seed = futures[future]
                current_run += 1
                run_metrics, elapsed = future.result()
                job_results[(bmark, alg, seed)] = (run_metrics, elapsed)
                if verbose:
                    rel_l2 = float(run_metrics.get("val_rel_l2", 1.0))
                    print(f"[{current_run}/{total_runs}] {alg:20s} {bmark:10s} seed={seed} done (rel_L2 = {rel_l2:.6f}, time = {elapsed:.2f}s)")
    else:
        for (bmark, alg, seed, out_dir) in job_specs:
            current_run += 1
            if verbose:
                print(f"[{current_run}/{total_runs}] Running {alg:15s} on {bmark:10s} (seed={seed})...", end="", flush=True)
            run_metrics, elapsed = _run_single_job(alg, out_dir, bmark, seed, config.n_steps, quick)
            job_results[(bmark, alg, seed)] = (run_metrics, elapsed)
            if verbose:
                rel_l2 = float(run_metrics.get("val_rel_l2", 1.0))
                print(f" done (rel_L2 = {rel_l2:.6f}, time = {elapsed:.2f}s)")

    for bmark in config.benchmarks:
        results["raw_runs"][bmark] = {}
        results["benchmark_summary"][bmark] = {}

        for alg in config.algorithms:
            if alg not in ALGORITHM_REGISTRY:
                print(f"Warning: algorithm '{alg}' not found in registry, skipping.")
                continue

            results["raw_runs"][bmark][alg] = []
            seed_errors = []
            seed_mses = []
            seed_times = []
            histories = []
            best_configs = []
            diversity_histories = []

            for seed in config.seeds:
                run_metrics, elapsed = job_results[(bmark, alg, seed)]

                rel_l2 = float(run_metrics.get("val_rel_l2", 1.0))
                mse = float(run_metrics.get("val_mse", 1.0))
                history = run_metrics.get("history", [rel_l2])
                best_cfg = run_metrics.get("config", {})
                diversity_history = run_metrics.get("diversity_history", [])

                run_entry = {
                    "seed": seed,
                    "val_rel_l2": rel_l2,
                    "val_mse": mse,
                    "runtime_sec": elapsed,
                    "history": history,
                    "best_config": best_cfg,
                    "diversity_history": diversity_history,
                }
                results["raw_runs"][bmark][alg].append(run_entry)

                seed_errors.append(rel_l2)
                seed_mses.append(mse)
                seed_times.append(elapsed)
                histories.append(history)
                best_configs.append(best_cfg)
                diversity_histories.append(diversity_history)

            # Aggregate statistics across seeds for this benchmark + algorithm
            min_len = min(len(h) for h in histories) if histories else 1
            trimmed_histories = np.array([h[:min_len] for h in histories], dtype=float)
            mean_history = list(np.mean(trimmed_histories, axis=0))

            best_seed_idx = int(np.argmin(seed_errors))

            # Aggregate real per-generation diversity (exploration) trajectories across seeds.
            # Fuzzy variants record this every generation/iteration via compute_population_diversity;
            # plain GA/PSO/ACO record it too (see src/hpo/{ga,pso,aco}.py). Only the pyswarm-library
            # PSO path (unused unless pyswarm is installed) has no per-iteration hook and yields [].
            valid_div_histories = [dh for dh in diversity_histories if dh]
            if valid_div_histories:
                div_min_len = min(len(dh) for dh in valid_div_histories)
                div_matrix = np.array(
                    [[step["diversity"] for step in dh[:div_min_len]] for dh in valid_div_histories],
                    dtype=float,
                )
                mean_diversity_trajectory = list(np.mean(div_matrix, axis=0))
                overall_mean_diversity = float(np.mean(div_matrix))
            else:
                mean_diversity_trajectory = []
                overall_mean_diversity = float("nan")

            results["benchmark_summary"][bmark][alg] = {
                "mean_rel_l2": float(np.mean(seed_errors)),
                "std_rel_l2": float(np.std(seed_errors)),
                "min_rel_l2": float(np.min(seed_errors)),
                "max_rel_l2": float(np.max(seed_errors)),
                "mean_mse": float(np.mean(seed_mses)),
                "mean_runtime_sec": float(np.mean(seed_times)),
                "mean_history": mean_history,
                "mean_diversity_trajectory": mean_diversity_trajectory,
                "overall_mean_diversity": overall_mean_diversity,
                "best_config": best_configs[best_seed_idx],
                "all_seed_rel_l2": seed_errors,
            }

    # Compute Global Friedman-style rankings and multi-criteria scores
    alg_ranks = {alg: [] for alg in config.algorithms}

    for bmark in config.benchmarks:
        summary = results["benchmark_summary"][bmark]
        # Sort algorithms by mean_rel_l2 ascending
        sorted_algs = sorted(summary.keys(), key=lambda a: summary[a]["mean_rel_l2"])
        for rank, alg in enumerate(sorted_algs, start=1):
            alg_ranks[alg].append(rank)

    global_rankings = {}
    for alg in config.algorithms:
        ranks = alg_ranks[alg]
        all_errs = [results["benchmark_summary"][bm][alg]["mean_rel_l2"] for bm in config.benchmarks]
        all_stds = [results["benchmark_summary"][bm][alg]["std_rel_l2"] for bm in config.benchmarks]
        all_runtimes = [results["benchmark_summary"][bm][alg]["mean_runtime_sec"] for bm in config.benchmarks]
        all_diversities = [
            results["benchmark_summary"][bm][alg]["overall_mean_diversity"] for bm in config.benchmarks
            if not np.isnan(results["benchmark_summary"][bm][alg]["overall_mean_diversity"])
        ]

        avg_rank = float(np.mean(ranks)) if ranks else 999.0
        overall_mean_error = float(np.mean(all_errs))
        overall_mean_std = float(np.mean(all_stds))
        overall_mean_runtime = float(np.mean(all_runtimes))
        overall_mean_diversity = float(np.mean(all_diversities)) if all_diversities else float("nan")

        global_rankings[alg] = {
            "average_rank": avg_rank,
            "overall_mean_rel_l2": overall_mean_error,
            "overall_mean_std": overall_mean_std,
            "overall_mean_runtime_sec": overall_mean_runtime,
            "overall_mean_diversity": overall_mean_diversity,
            "benchmark_ranks": {bm: alg_ranks[alg][i] for i, bm in enumerate(config.benchmarks)},
        }

    # Sort algorithms by average rank
    sorted_overall = sorted(global_rankings.keys(), key=lambda a: global_rankings[a]["average_rank"])
    results["overall_rankings"] = {a: global_rankings[a] for a in sorted_overall}

    # Save complete benchmark results JSON
    save_json(os.path.join(config.output_dir, "hpo_comparison_results.json"), results)
    return results


if __name__ == "__main__":
    try:
        from .report_generator import generate_all_plots, generate_markdown_report
    except (ImportError, ValueError):
        from hpo.report_generator import generate_all_plots, generate_markdown_report

    cfg = ExperimentConfig()
    results = run_experiment_grid(cfg, quick=False, verbose=True)
    plots_dir = os.path.join(cfg.output_dir, "plots")
    plot_files = generate_all_plots(results, plots_dir)
    report_file = os.path.join(cfg.output_dir, "HPO_INVESTIGATION_REPORT.md")
    generate_markdown_report(results, plot_files, report_file)
    print(f"\n[+] Full Investigation Report Generated: {report_file}")
