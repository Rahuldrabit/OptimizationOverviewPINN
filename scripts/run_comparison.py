"""Controlled PDE-Robust-DE comparison and publication plot generation.

The comparison uses only methods needed to evaluate PDE-Robust-DE: adaptive
DE, fixed-parameter DE, random search, GA, PSO, GSA, ACO, a two-stage baseline,
and a repeated default PINN configuration. Every method receives the same
candidate budget.

Example:
    python scripts/run_comparison.py --steps 1200 --evals 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from hpo.pde_robust_optimizer import _decode_solution, run_pde_robust_opt
from hpo.search_space import SearchSpace, decode_solution
from training.pinn_trainer import TrainConfig, train_pinn
from utils import ensure_dir

METHODS = [
    "PDE-Robust-DE", "Fixed DE", "Random Search", "GA", "PSO", "GSA", "ACO",
    "Two-Stage Baseline", "Default PINN",
]
COLORS = {
    "PDE-Robust-DE": "#0b6e4f",
    "Fixed DE": "#2f6690",
    "Random Search": "#d17a22",
    "Default PINN": "#777777",
    "GA": "#8e44ad",
    "PSO": "#c0392b",
    "GSA": "#16a085",
    "ACO": "#f39c12",
    "Two-Stage Baseline": "#34495e",
}


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _random_search(
    output_dir: str, benchmark: str, seed: int, evals: int, steps: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=steps, benchmark_type=benchmark)
    lower, upper = space.get_bounds()
    history: list[float] = []
    best = float("inf")
    best_config: dict[str, Any] = {}
    started = time.perf_counter()
    for _ in range(evals):
        candidate = lower + (upper - lower) * rng.random(len(lower))
        config = decode_solution(candidate, space, base)
        metrics = train_pinn(config)
        error = float(metrics["val_rel_l2"])
        best = min(best, error)
        history.append(best)
        if error <= best:
            best_config = metrics["config"]
    return {
        "final_error": history[-1],
        "history": history,
        "n_evaluations": len(history),
        "runtime_sec": time.perf_counter() - started,
        "config": best_config,
    }


def _default_pinn(
    benchmark: str, seed: int, evals: int, steps: int
) -> dict[str, Any]:
    base = TrainConfig(seed=seed, n_steps=steps, benchmark_type=benchmark)
    history: list[float] = []
    best = float("inf")
    started = time.perf_counter()
    for _ in range(evals):
        metrics = train_pinn(base)
        best = min(best, float(metrics["val_rel_l2"]))
        history.append(best)
    return {
        "final_error": history[-1],
        "history": history,
        "n_evaluations": len(history),
        "runtime_sec": time.perf_counter() - started,
        "config": metrics["config"],
    }


def _generic_search(
    benchmark: str, seed: int, evals: int, steps: int, method: str
) -> dict[str, Any]:
    """Run a lightweight baseline with exact candidate-level accounting."""
    rng = np.random.default_rng(seed)
    space = SearchSpace()
    base = TrainConfig(seed=seed, n_steps=steps, benchmark_type=benchmark)
    lower, upper = space.get_bounds()
    dimension = len(lower)
    history: list[float] = []
    best_error = float("inf")
    best_config: dict[str, Any] = {}
    started = time.perf_counter()

    def evaluate(candidate: np.ndarray) -> float:
        nonlocal best_error, best_config
        config = decode_solution(candidate, space, base)
        metrics = train_pinn(config)
        error = float(metrics["val_rel_l2"])
        if error < best_error:
            best_error = error
            best_config = metrics["config"]
        history.append(best_error)
        return error

    population_size = min(10, evals)
    if population_size < 4:
        raise ValueError("evals must be at least 4 for the population baselines")

    if method == "GA":
        population = lower + (upper - lower) * rng.random((population_size, dimension))
        fitness = np.array([evaluate(candidate) for candidate in population])
        while len(history) < evals:
            children = [population[int(np.argmin(fitness))].copy()]
            while len(children) < population_size:
                indices = rng.choice(population_size, size=3, replace=False)
                parent_a = population[indices[np.argmin(fitness[indices])]]
                indices = rng.choice(population_size, size=3, replace=False)
                parent_b = population[indices[np.argmin(fitness[indices])]]
                point = int(rng.integers(1, dimension))
                child = np.concatenate([parent_a[:point], parent_b[point:]])
                mutation = rng.random(dimension) < 0.2
                child[mutation] = lower[mutation] + (upper[mutation] - lower[mutation]) * rng.random(np.sum(mutation))
                children.append(np.clip(child, lower, upper))
            population = np.asarray(children)
            for index, candidate in enumerate(population):
                if len(history) >= evals:
                    break
                fitness[index] = evaluate(candidate)

    elif method == "PSO":
        positions = lower + (upper - lower) * rng.random((population_size, dimension))
        velocities = rng.uniform(-0.25, 0.25, (population_size, dimension)) * (upper - lower)
        personal = positions.copy()
        personal_fit = np.array([evaluate(candidate) for candidate in positions])
        best_index = int(np.argmin(personal_fit))
        global_best = personal[best_index].copy()
        while len(history) < evals:
            r1 = rng.random((population_size, dimension))
            r2 = rng.random((population_size, dimension))
            velocities = 0.7 * velocities + 1.4 * r1 * (personal - positions) + 1.6 * r2 * (global_best - positions)
            velocities = np.clip(velocities, -0.25 * (upper - lower), 0.25 * (upper - lower))
            positions = np.clip(positions + velocities, lower, upper)
            for index, candidate in enumerate(positions):
                if len(history) >= evals:
                    break
                fitness = evaluate(candidate)
                if fitness < personal_fit[index]:
                    personal_fit[index] = fitness
                    personal[index] = candidate.copy()
            global_best = personal[int(np.argmin(personal_fit))].copy()

    elif method == "GSA":
        positions = lower + (upper - lower) * rng.random((population_size, dimension))
        fitness = np.array([evaluate(candidate) for candidate in positions])
        while len(history) < evals:
            best_index = int(np.argmin(fitness))
            attraction = positions[best_index] - positions
            positions = np.clip(positions + rng.random(positions.shape) * 0.2 * attraction, lower, upper)
            for index, candidate in enumerate(positions):
                if len(history) >= evals:
                    break
                fitness[index] = evaluate(candidate)

    elif method == "ACO":
        archive = lower + (upper - lower) * rng.random((population_size, dimension))
        fitness = np.array([evaluate(candidate) for candidate in archive])
        while len(history) < evals:
            order = np.argsort(fitness)
            archive, fitness = archive[order], fitness[order]
            weights = np.exp(-np.arange(population_size) / max(1.0, population_size * 0.35))
            weights /= weights.sum()
            spread = np.std(archive, axis=0) + 1e-8
            candidates = []
            for _ in range(population_size):
                if len(history) >= evals:
                    break
                index = int(rng.choice(population_size, p=weights))
                candidates.append(np.clip(rng.normal(archive[index], spread), lower, upper))
            for candidate in candidates:
                fitness_value = evaluate(candidate)
                archive = np.vstack([archive, candidate])
                fitness = np.append(fitness, fitness_value)
            order = np.argsort(fitness)[:population_size]
            archive, fitness = archive[order], fitness[order]

    elif method == "Two-Stage Baseline":
        stage_one = max(1, int(evals * 0.7))
        candidates = lower + (upper - lower) * rng.random((population_size, dimension))
        fitness = np.full(population_size, np.inf)
        for index in range(stage_one):
            candidate = candidates[index % population_size]
            fitness[index % population_size] = evaluate(candidate)
        elites = candidates[np.argsort(fitness)[:max(1, min(3, population_size))]]
        while len(history) < evals:
            center = elites[int(rng.integers(len(elites)))]
            candidate = np.clip(center + rng.normal(0.0, 0.08, dimension) * (upper - lower), lower, upper)
            evaluate(candidate)
    else:
        raise ValueError(f"Unknown generic method: {method}")

    return {
        "final_error": history[-1],
        "history": history,
        "n_evaluations": len(history),
        "runtime_sec": time.perf_counter() - started,
        "config": best_config,
    }


def _de_run(
    output_dir: str,
    benchmark: str,
    seed: int,
    evals: int,
    steps: int,
    adaptive: bool,
    bounce_back: bool,
) -> dict[str, Any]:
    population = min(10, evals)
    if population < 4 or evals % population:
        raise ValueError("evals must be divisible by a population size of at least 4")
    generations = (evals // population) - 1
    metrics = run_pde_robust_opt(
        out_dir=output_dir,
        benchmark_type=benchmark,
        seed=seed,
        n_generations=generations,
        sol_per_pop=population,
        n_steps=steps,
        adaptive=adaptive,
        bounce_back=bounce_back,
    )
    if metrics["n_evaluations"] != evals:
        raise RuntimeError(
            f"Expected {evals} evaluations, got {metrics['n_evaluations']}"
        )
    return {
        "final_error": float(metrics["history"][-1]),
        "history": metrics["history"],
        "n_evaluations": metrics["n_evaluations"],
        "runtime_sec": 0.0,
        "config": metrics["config"],
    }


def run_method(
    method: str,
    output_dir: str,
    benchmark: str,
    seed: int,
    evals: int,
    steps: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    if method == "PDE-Robust-DE":
        result = _de_run(output_dir, benchmark, seed, evals, steps, True, True)
    elif method == "Fixed DE":
        result = _de_run(output_dir, benchmark, seed, evals, steps, False, False)
    elif method == "Random Search":
        result = _random_search(output_dir, benchmark, seed, evals, steps)
    elif method == "Default PINN":
        result = _default_pinn(benchmark, seed, evals, steps)
    elif method in {"GA", "PSO", "GSA", "ACO", "Two-Stage Baseline"}:
        result = _generic_search(benchmark, seed, evals, steps, method)
    else:
        raise ValueError(f"Unknown comparison method: {method}")
    result["runtime_sec"] = time.perf_counter() - started
    return result


def run_ablation(
    output_dir: str, benchmarks: list[str], seeds: list[int], evals: int, steps: int
) -> dict[str, list[dict[str, Any]]]:
    variants = {
        "Adaptive + Bounce-back": (True, True),
        "Adaptive + Clipping": (True, False),
        "Fixed + Clipping": (False, False),
    }
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    for benchmark in benchmarks:
        for seed in seeds:
            for name, (adaptive, bounce_back) in variants.items():
                variant_dir = os.path.join(output_dir, "ablation", benchmark, name.lower().replace(" ", "_"), f"seed_{seed}")
                started = time.perf_counter()
                result = _de_run(variant_dir, benchmark, seed, evals, steps, adaptive, bounce_back)
                result.update({"benchmark": benchmark, "seed": seed, "runtime_sec": time.perf_counter() - started})
                results[name].append(result)
    return results


def _group(results: list[dict[str, Any]], benchmark: str, method: str) -> list[dict[str, Any]]:
    return [row for row in results if row["benchmark"] == benchmark and row["method"] == method]


def plot_comparison(data: dict[str, Any], output_dir: str, threshold: float) -> None:
    plot_dir = os.path.join(output_dir, "plots")
    ensure_dir(plot_dir)
    benchmarks = data["metadata"]["benchmarks"]
    results = data["results"]

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(4 * len(benchmarks), 5), squeeze=False)
    for axis, benchmark in zip(axes[0], benchmarks):
        values = [[row["final_error"] for row in _group(results, benchmark, method)] for method in METHODS]
        axis.boxplot(values, tick_labels=METHODS, showfliers=True)
        axis.set_yscale("log")
        axis.set_title(benchmark.upper())
        axis.set_ylabel("Final validation relative L2 error")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Final accuracy distribution across seeds")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "final_error_distributions.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(4 * len(benchmarks), 5), squeeze=False)
    for axis, benchmark in zip(axes[0], benchmarks):
        for method in METHODS:
            histories = np.array([row["history"] for row in _group(results, benchmark, method)])
            x = np.arange(1, histories.shape[1] + 1)
            median = np.median(histories, axis=0)
            lower, upper = np.percentile(histories, [25, 75], axis=0)
            axis.plot(x, median, label=method, color=COLORS[method])
            axis.fill_between(x, lower, upper, color=COLORS[method], alpha=0.12)
        axis.set_yscale("log")
        axis.set_title(benchmark.upper())
        axis.set_xlabel("Candidate evaluations")
        axis.set_ylabel("Best-so-far relative L2 error")
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Evaluation-based convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "convergence_curves.png"), dpi=180)
    plt.close(fig)

    matrix = np.array([
        [np.median([row["final_error"] for row in _group(results, benchmark, method)]) for benchmark in benchmarks]
        for method in METHODS
    ])
    fig, axis = plt.subplots(figsize=(8, 4.5))
    image = axis.imshow(np.log10(np.maximum(matrix, 1e-12)), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(benchmarks)), [name.upper() for name in benchmarks])
    axis.set_yticks(range(len(METHODS)), METHODS)
    for i in range(len(METHODS)):
        for j in range(len(benchmarks)):
            axis.text(j, i, f"{matrix[i, j]:.2e}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axis, label="log10 median relative L2 error")
    axis.set_title("Median final error heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "accuracy_heatmap.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(4 * len(benchmarks), 5), squeeze=False)
    for axis, benchmark in zip(axes[0], benchmarks):
        for method in METHODS:
            histories = np.array([row["history"] for row in _group(results, benchmark, method)])
            success = np.mean(histories <= threshold, axis=0)
            axis.plot(np.arange(1, len(success) + 1), success, label=method, color=COLORS[method])
        axis.set_ylim(0, 1.05)
        axis.set_title(benchmark.upper())
        axis.set_xlabel("Candidate evaluations")
        axis.set_ylabel(f"Fraction reaching error <= {threshold:g}")
        axis.grid(True, linestyle=":", alpha=0.5)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Success rate by evaluation budget")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "success_rate.png"), dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        rows = [row for row in results if row["method"] == method]
        x = np.mean([row["runtime_sec"] for row in rows])
        y = np.median([row["final_error"] for row in rows])
        axis.scatter(x, y, s=100, color=COLORS[method], label=method)
        axis.annotate(method, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8)
    axis.set_yscale("log")
    axis.set_xlabel("Mean runtime per run (seconds)")
    axis.set_ylabel("Median final relative L2 error")
    axis.set_title("Runtime-accuracy trade-off")
    axis.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "runtime_accuracy_pareto.png"), dpi=180)
    plt.close(fig)

    ablation = data.get("ablation", {})
    if ablation:
        fig, axis = plt.subplots(figsize=(8, 5))
        names = list(ablation)
        values = [
            [row["final_error"] for row in ablation[name]]
            for name in names
        ]
        axis.boxplot(values, tick_labels=names, showfliers=True)
        axis.set_yscale("log")
        axis.set_ylabel("Final validation relative L2 error")
        axis.set_title("PDE-Robust-DE boundary/adaptation ablation")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "pde_robust_de_ablation.png"), dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled PDE-Robust-DE comparisons")
    parser.add_argument("--benchmarks", nargs="+", default=["ode", "heat", "burgers", "wave"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--evals", type=int, default=80)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--out-dir", default="outputs/pde_robust_de_comparison")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    rows: list[dict[str, Any]] = []
    for benchmark in args.benchmarks:
        for method in METHODS:
            for seed in args.seeds:
                method_dir = os.path.join(args.out_dir, "runs", benchmark, method.lower().replace(" ", "_"), f"seed_{seed}")
                print(f"{benchmark.upper():8s} {method:16s} seed={seed}")
                result = run_method(method, method_dir, benchmark, seed, args.evals, args.steps)
                rows.append({"benchmark": benchmark, "method": method, "seed": seed, **result})

    data: dict[str, Any] = {
        "metadata": {
            "benchmarks": args.benchmarks,
            "methods": METHODS,
            "seeds": args.seeds,
            "evaluations_per_run": args.evals,
            "training_steps_per_candidate": args.steps,
            "threshold": args.threshold,
        },
        "results": rows,
    }
    if not args.skip_ablation:
        data["ablation"] = run_ablation(args.out_dir, args.benchmarks, args.seeds, args.evals, args.steps)
    _save_json(os.path.join(args.out_dir, "comparison_results.json"), data)
    plot_comparison(data, args.out_dir, args.threshold)
    print(f"Saved comparison results and plots to {args.out_dir}")


if __name__ == "__main__":
    main()
