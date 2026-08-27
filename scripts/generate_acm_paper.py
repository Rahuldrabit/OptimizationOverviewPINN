"""Generate an ACM sigconf (two-column) paper comparing GA, PSO, ACO, Fuzzy-GA,
Fuzzy-PSO, and Fuzzy-ACO for PINN hyperparameter optimization, for NSYS 2026.

Reads outputs/nsys2026/hpo_comparison_results.json (produced by
scripts/run_nsys2026_manuscript.py) and reports only numbers that are actually
in that file - real per-generation convergence and diversity/exploration
trajectories (src/hpo/{ga,pso,aco,fuzzy_ga,fuzzy_pso,fuzzy_aco}.py), not
assumed or hardcoded scores. See MEMORY note "paper-not-publishable": do not
add claims here that the underlying JSON does not support.

Usage:
    python scripts/generate_acm_paper.py --results-dir outputs/nsys2026
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

ALGORITHMS = ["GA", "PSO", "ACO", "Fuzzy-GA", "Fuzzy-PSO", "Fuzzy-ACO"]


def load_results(results_dir: str) -> dict[str, Any]:
    results_file = os.path.join(results_dir, "hpo_comparison_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            f"Run scripts/run_nsys2026_manuscript.py first to generate it."
        )
    with open(results_file, "r") as f:
        return json.load(f)


def compute_algo_stats(results: dict[str, Any], algorithms: list[str]) -> dict[str, Any]:
    """Aggregate mean/std error, runtime, and mean diversity per algorithm, across
    all benchmarks and seeds present in the results file. All numbers are computed
    directly from raw_runs - nothing here is a hardcoded or assumed constant."""
    raw_runs = results.get("raw_runs", {})
    benchmarks = results.get("metadata", {}).get("benchmarks", [])
    stats: dict[str, Any] = {}

    for algo in algorithms:
        errs, times, diversities, n_generations = [], [], [], []
        per_benchmark: dict[str, list[float]] = {}

        for bmark in benchmarks:
            runs = raw_runs.get(bmark, {}).get(algo, [])
            bmark_errs = [r["val_rel_l2"] for r in runs]
            per_benchmark[bmark] = bmark_errs
            for r in runs:
                errs.append(r["val_rel_l2"])
                times.append(r["runtime_sec"])
                dh = r.get("diversity_history", [])
                if dh:
                    diversities.extend(step["diversity"] for step in dh)
                    n_generations.append(len(dh))

        if not errs:
            continue

        stats[algo] = {
            "mean_l2": float(np.mean(errs)),
            "std_l2": float(np.std(errs)),
            "min_l2": float(np.min(errs)),
            "max_l2": float(np.max(errs)),
            "mean_time": float(np.mean(times)),
            "mean_diversity": float(np.mean(diversities)) if diversities else float("nan"),
            "mean_generations": float(np.mean(n_generations)) if n_generations else float("nan"),
            "n_runs": len(errs),
            "per_benchmark": per_benchmark,
        }

    return stats


def significance_note(stats: dict[str, Any], n_seeds: int) -> str:
    """Best-effort paired significance check between the top-ranked and the plain-GA
    baseline. Honest about statistical power instead of asserting significance from
    a handful of seeds (see MEMORY: prior draft claimed significance from 2 seeds)."""
    if n_seeds < 5:
        return (
            f"With only {n_seeds} random seed(s) per (algorithm, benchmark) cell, this study is "
            r"\emph{not} statistically powered for a formal significance test (e.g., Wilcoxon "
            r"signed-rank or Friedman); rank differences below should be read as descriptive, "
            r"not as evidence of a significant effect. A follow-up with $\geq$10 seeds is needed "
            "before any claim of statistical significance."
        )
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return "scipy was not available to compute a formal significance test; ranks are descriptive only."

    sorted_algs = sorted(stats.items(), key=lambda kv: kv[1]["mean_l2"])
    best_alg = sorted_algs[0][0]
    if best_alg == "GA" or "GA" not in stats:
        return "Best-ranked algorithm is the GA baseline itself; no separate significance test applies."
    return (
        f"A Wilcoxon signed-rank test between {best_alg} and the GA baseline "
        "would require paired per-seed samples on matching benchmarks; see raw_runs in the "
        "results JSON for the underlying paired values before citing a p-value."
    )


def generate_latex_paper(results: dict[str, Any], stats: dict[str, Any], output_file: str, figures_rel: str) -> None:
    benchmarks = results.get("metadata", {}).get("benchmarks", [])
    seeds = results.get("metadata", {}).get("seeds", [])

    sorted_algos = sorted(stats.items(), key=lambda kv: kv[1]["mean_l2"])
    best_alg, best_stats = sorted_algos[0]
    worst_alg, worst_stats = sorted_algos[-1]

    fuzzy_algs = [a for a, _ in sorted_algos if "Fuzzy" in a]
    plain_algs = [a for a, _ in sorted_algos if "Fuzzy" not in a]
    best_fuzzy = fuzzy_algs[0] if fuzzy_algs else None
    best_plain = plain_algs[0] if plain_algs else None

    sig_note = significance_note(stats, len(seeds))

    latex: list[str] = []
    latex.append(r"\documentclass[sigconf]{acmart}")
    latex.append(r"""
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{}
\pagestyle{plain}

\title{An Empirical Comparison of Genetic, Particle Swarm, Ant Colony, and Fuzzy-Adaptive Metaheuristics for Physics-Informed Neural Network Hyperparameter Optimization}

\author{Rahul Drabit Chowdhury}
\affiliation{\institution{Independent Research}\country{}}

\begin{document}
\begin{abstract}""")

    latex.append(
        f"We empirically compare six metaheuristic hyperparameter optimizers -- Genetic Algorithm (GA), "
        f"Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO/ACOR), and their Mamdani "
        f"fuzzy-adaptive variants (Fuzzy-GA, Fuzzy-PSO, Fuzzy-ACO) -- for tuning Physics-Informed Neural "
        f"Networks (PINNs) across {len(benchmarks)} PDE benchmarks ({', '.join(b.upper() for b in benchmarks)}) "
        f"with {len(seeds)} random seed(s) per configuration. Unlike prior comparisons that report only final "
        f"error, we log real per-generation population/swarm/archive diversity for every algorithm "
        f"(not only the fuzzy variants) and use it to characterize exploration-exploitation behavior "
        f"directly, rather than assuming it from algorithm category. "
        f"The best-performing method in this run, {best_alg}, reaches a mean relative "
        f"$L_2$ error of {best_stats['mean_l2']:.6f} (min {best_stats['min_l2']:.6f}) versus "
        f"{worst_stats['mean_l2']:.6f} for the weakest, {worst_alg}. {sig_note}"
    )
    latex.append(r"\end{abstract}")

    latex.append(r"""
\maketitle

\section{Introduction}
Physics-Informed Neural Networks (PINNs)~\cite{raissi2019physics,karniadakis2021physics} embed PDE
residuals directly into the training loss, but their accuracy is highly sensitive to hyperparameters
(architecture, activation, optimizer, learning rate, and physics/initial-condition loss weights).
Recent work has explored evolutionary strategies for PINN hyperparameter optimization (HPO), including
a two-stage evolutionary approach~\cite{buzaev2026evolutionary} and evolutionary PINNs for inverse
modeling~\cite{jasim2026evopinn}. This paper reports a controlled, same-codebase comparison of six
classical and fuzzy-adaptive metaheuristics under an identical search space and training budget, with
particular attention to two properties that prior comparisons typically assume rather than measure:
population/swarm/archive \emph{diversity} across generations, and how that diversity trades off against
convergence speed.

\section{Methodology}

\subsection{Algorithms}
\begin{itemize}
    \item \textbf{GA}~\cite{holland1992genetic}: tournament selection, single-point crossover, elitist replacement.
    \item \textbf{PSO}~\cite{kennedy1995particle}: inertia-weighted velocity update with cognitive/social terms.
    \item \textbf{ACO / ACOR}~\cite{dorigo2006ant,socha2008ant}: continuous ant colony optimization with a Gaussian-kernel-weighted solution archive.
    \item \textbf{Fuzzy-GA / Fuzzy-PSO / Fuzzy-ACO}: each classical algorithm augmented with a Mamdani fuzzy inference controller~\cite{mamdani1975experiment} that observes measured population diversity, fitness improvement rate, and search progress each generation, and outputs exploration/exploitation weights that adapt mutation rate, inertia, or archive spread accordingly.
\end{itemize}

\subsection{Diversity / Exploration Measurement}
For every algorithm (not only the fuzzy variants), we compute a normalized diversity score each
generation/iteration as the mean Euclidean distance of the population/swarm/archive to its own
centroid in the normalized search space, divided by the theoretical maximum spread of a unit
hypercube of the same dimensionality. This gives a directly comparable, measured exploration signal
across all six algorithms instead of an assumed category-based bonus.

\subsection{Search Space}
All algorithms optimize the same 8-dimensional space: hidden layers, hidden width, activation
(tanh / sine / swish), optimizer (Adam / AdamW / L-BFGS), learning rate (log-scale), physics loss
weight, initial-condition loss weight, and number of collocation points.

\subsection{Benchmarks}
""")
    latex.append(
        f"Evaluation covers {len(benchmarks)} PDE benchmarks with real PyTorch PINN training: "
        f"{', '.join(b.upper() for b in benchmarks)}. Other PDE families (Allen-Cahn, "
        "reaction-diffusion, Navier-Stokes, Helmholtz) are deliberately excluded from this paper "
        "because this codebase does not yet have a verified, non-placeholder trainer for them."
    )

    latex.append(r"""
\section{Results}

\subsection{Overall Ranking}
\begin{table}[h]
\centering
\small
\caption{Aggregate performance across all benchmarks and seeds (mean over """ + str(len(benchmarks)) + r""" benchmarks $\times$ """ + str(len(seeds)) + r""" seed(s)). Diversity is the measured mean normalized population/swarm/archive spread across all recorded generations.}
\begin{tabular}{lccc}
\toprule
\textbf{Algorithm} & \textbf{Mean $L_2$} & \textbf{Std $L_2$} & \textbf{Mean Diversity} \\
\midrule
""")
    for algo, s in sorted_algos:
        div_str = f"{s['mean_diversity']:.3f}" if not np.isnan(s["mean_diversity"]) else "n/a"
        latex.append(f"{algo} & {s['mean_l2']:.6f} & {s['std_l2']:.6f} & {div_str} \\\\")
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    latex.append(r"\subsection{Convergence and Diversity Trajectories}")
    latex.append(
        r"Figure~\ref{fig:convergence} shows mean per-generation validation error, and "
        r"Figure~\ref{fig:diversity} shows the corresponding measured diversity trajectory for the "
        r"same runs. A useful search dynamic is visible where diversity is high in early generations "
        r"and declines as the population/swarm/archive converges; algorithms that collapse diversity "
        r"too early risk premature convergence, while those that sustain it too long converge slowly."
    )
    latex.append(r"""
\begin{figure}[h]
\centering
\includegraphics[width=\linewidth]{""" + figures_rel + r"""/convergence_comparison.png}
\caption{Mean validation relative $L_2$ error per generation/iteration, by benchmark.}
\label{fig:convergence}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=\linewidth]{""" + figures_rel + r"""/diversity_exploration_trajectories.png}
\caption{Measured population/swarm/archive diversity per generation/iteration, by benchmark.}
\label{fig:diversity}
\end{figure}
""")

    if best_plain and best_fuzzy:
        plain_l2 = stats[best_plain]["mean_l2"]
        fuzzy_l2 = stats[best_fuzzy]["mean_l2"]
        direction = "lower" if fuzzy_l2 < plain_l2 else "higher"
        latex.append(
            f"\\subsection{{Fuzzy-Adaptive vs.\\ Classical Variants}}\n"
            f"The best classical (non-fuzzy) algorithm in this run is {best_plain} "
            f"(mean $L_2$ = {plain_l2:.6f}); the best fuzzy-adaptive variant is {best_fuzzy} "
            f"(mean $L_2$ = {fuzzy_l2:.6f}), a {direction} mean error. "
        )

    latex.append(r"\subsection{Comparison with Recent PINN-HPO Literature}")
    latex.append(
        r"Buzaev et al.~\cite{buzaev2026evolutionary} propose a two-stage evolutionary strategy for "
        r"PINN HPO; Jasim et al.~\cite{jasim2026evopinn} apply evolutionary PINNs to inverse geotechnical "
        r"and structural modeling. Both use PDE benchmarks and search spaces that differ from this study's, "
        r"so we report our results alongside theirs qualitatively rather than asserting a numeric "
        r"head-to-head comparison; a fair quantitative comparison would require re-running their method "
        r"in this codebase under matched search-space and training-step budgets."
    )

    latex.append(r"\section{Threats to Validity}")
    latex.append(
        f"(1) {sig_note} "
        "(2) Training runs on CPU with a fixed step budget; results may shift under GPU training or "
        "a larger step budget. (3) Only four PDE benchmarks with verified real trainers are included; "
        "generalization to other PDE families is untested here."
    )

    latex.append(r"""
\section{Reproducibility}
All code is available at \texttt{https://github.com/Rahuldrabit/OptimizationOverviewPINN}. Every number
in this paper is generated directly from \texttt{outputs/nsys2026/hpo\_comparison\_results.json},
produced by \texttt{scripts/run\_nsys2026\_manuscript.py}.

\bibliographystyle{ACM-Reference-Format}
\bibliography{references}

\end{document}
""")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))

    print(f"[OK] LaTeX paper generated: {output_file}")


def generate_comparison_csv(stats: dict[str, Any], output_file: str) -> None:
    import csv

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Mean L2", "Std L2", "Min L2", "Max L2", "Mean Runtime (s)", "Mean Diversity", "N Runs"])
        for algo, s in sorted(stats.items(), key=lambda kv: kv[1]["mean_l2"]):
            writer.writerow([
                algo, f"{s['mean_l2']:.6f}", f"{s['std_l2']:.6f}", f"{s['min_l2']:.6f}", f"{s['max_l2']:.6f}",
                f"{s['mean_time']:.2f}",
                f"{s['mean_diversity']:.3f}" if not np.isnan(s["mean_diversity"]) else "n/a",
                s["n_runs"],
            ])

    print(f"[OK] Comparison CSV generated: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ACM sigconf paper from NSYS 2026 HPO results")
    parser.add_argument("--results-dir", default="outputs/nsys2026")
    parser.add_argument("--output-paper", default=None, help="Default: <results-dir>/paper/nsys2026_paper.tex")
    parser.add_argument("--output-table", default=None, help="Default: <results-dir>/paper/nsys2026_comparison.csv")
    args = parser.parse_args()

    output_paper = args.output_paper or os.path.join(args.results_dir, "paper", "nsys2026_paper.tex")
    output_table = args.output_table or os.path.join(args.results_dir, "paper", "nsys2026_comparison.csv")
    os.makedirs(os.path.dirname(output_paper), exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING ACM SIGCONF PAPER FOR NSYS 2026")
    print("=" * 70 + "\n")

    results = load_results(args.results_dir)
    stats = compute_algo_stats(results, ALGORITHMS)

    if not stats:
        print("[ERROR] No matching algorithm results found in raw_runs. "
              "Did you run scripts/run_nsys2026_manuscript.py with GA/PSO/ACO/Fuzzy-* ?")
        return

    bib_src = project_root / "paper" / "references.bib"
    bib_dst = os.path.join(os.path.dirname(output_paper), "references.bib")
    if bib_src.exists():
        shutil.copyfile(bib_src, bib_dst)

    generate_latex_paper(results, stats, output_paper, figures_rel="../plots")
    generate_comparison_csv(stats, output_table)

    print("\n" + "=" * 70)
    print("ACM PAPER GENERATION COMPLETE")
    print(f"LaTeX:      {os.path.abspath(output_paper)}")
    print(f"References: {os.path.abspath(bib_dst)}")
    print(f"CSV:        {os.path.abspath(output_table)}")
    print("\nNext steps:")
    print(f"1. cd {os.path.dirname(output_paper)} && pdflatex nsys2026_paper.tex && bibtex nsys2026_paper && pdflatex nsys2026_paper.tex && pdflatex nsys2026_paper.tex")
    print("2. Review the PDF - check every number against hpo_comparison_results.json before submitting")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
