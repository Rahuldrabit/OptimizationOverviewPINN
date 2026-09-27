"""Statistical analysis for raw seed-level PINN HPO results."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


def error_value(run: dict[str, Any]) -> float:
    return float(run.get("val_rel_l2", run["final_error"]))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.ones(len(p_values), dtype=float)
    for position, original_index in enumerate(order):
        adjusted[original_index] = min(1.0, (len(p_values) - position) * p_values[original_index])
    for position in range(1, len(order)):
        adjusted[order[position]] = max(adjusted[order[position]], adjusted[order[position - 1]])
    return adjusted.tolist()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    if len(values) < 2:
        return [float(values[0]), float(values[0])]
    samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def analyze(results: dict[str, Any], reference: str) -> dict[str, Any]:
    rng = np.random.default_rng(20260927)
    report: dict[str, Any] = {"reference": reference, "benchmarks": {}}
    algorithms = results["metadata"]["algorithms"]

    for benchmark, benchmark_data in results["raw_runs"].items():
        errors = {
            algorithm: np.array([error_value(run) for run in benchmark_data[algorithm]], dtype=float)
            for algorithm in algorithms
        }
        try:
            friedman_stat, friedman_p = map(float, friedmanchisquare(*(errors[a] for a in algorithms)))
        except ValueError:
            friedman_stat, friedman_p = float("nan"), float("nan")

        comparisons = []
        p_values = []
        for algorithm in algorithms:
            if algorithm == reference:
                continue
            difference = errors[algorithm] - errors[reference]
            try:
                statistic, p_value = map(float, wilcoxon(errors[reference], errors[algorithm], zero_method="wilcox"))
            except ValueError:
                statistic, p_value = float("nan"), 1.0
            p_values.append(p_value)
            comparisons.append({
                "algorithm": algorithm,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "mean_difference_comparator_minus_reference": float(np.mean(difference)),
                "bootstrap_95ci_difference": bootstrap_ci(difference, rng),
            })
        for comparison, adjusted in zip(comparisons, holm_adjust(p_values)):
            comparison["holm_adjusted_p_value"] = adjusted

        report["benchmarks"][benchmark] = {
            "n_seeds": len(next(iter(errors.values()))),
            "friedman_statistic": friedman_stat,
            "friedman_p_value": friedman_p,
            "algorithms": {
                algorithm: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "bootstrap_95ci_mean": bootstrap_ci(values, rng),
                    "values": values.tolist(),
                }
                for algorithm, values in errors.items()
            },
            "paired_comparisons": comparisons,
        }
    return report


def write_markdown(report: dict[str, Any], output_file: str) -> None:
    lines = ["# Seed-Level Statistical Analysis", "", f"Reference algorithm: **{report['reference']}**", "", "Positive differences mean the comparator had higher error. Wilcoxon p-values are Holm-adjusted within each benchmark.", ""]
    for benchmark, data in report["benchmarks"].items():
        lines.extend([f"## {benchmark.upper()}", "", f"Seeds: {data['n_seeds']}", f"Friedman statistic: `{data['friedman_statistic']:.6g}`; p-value: `{data['friedman_p_value']:.6g}`", "", "| Algorithm | Mean | Median | Sample SD | 95% CI for mean |", "| :--- | ---: | ---: | ---: | :--- |"])
        for algorithm, stats in data["algorithms"].items():
            ci = stats["bootstrap_95ci_mean"]
            lines.append(f"| {algorithm} | {stats['mean']:.6g} | {stats['median']:.6g} | {stats['std_sample']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] |")
        lines.extend(["", "| Comparator | Mean difference | 95% CI difference | Holm-adjusted p-value |", "| :--- | ---: | :--- | ---: |"])
        for comparison in data["paired_comparisons"]:
            ci = comparison["bootstrap_95ci_difference"]
            lines.append(f"| {comparison['algorithm']} | {comparison['mean_difference_comparator_minus_reference']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {comparison['holm_adjusted_p_value']:.6g} |")
        lines.append("")
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw seed-level HPO results")
    parser.add_argument("--results", required=True)
    parser.add_argument("--reference", default="PDE-Robust-DE")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as handle:
        results = json.load(handle)
    output_dir = args.out_dir or os.path.dirname(args.results) or "."
    os.makedirs(output_dir, exist_ok=True)
    report = analyze(results, args.reference)
    with open(os.path.join(output_dir, "statistical_analysis.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_markdown(report, os.path.join(output_dir, "STATISTICAL_ANALYSIS.md"))
    print(f"Saved statistical analysis to {output_dir}")


if __name__ == "__main__":
    main()
