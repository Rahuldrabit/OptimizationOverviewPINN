"""Hyperparameter Optimization (HPO) Package for PINNs.

Includes Standalone Optimizers (GA, PSO, ACO, GSA), Fuzzy-Enhanced Variants
(Fuzzy-GA, Fuzzy-PSO, Fuzzy-ACO), and Hybrids (GA-PSO, PSO-GSA, ACO-GA).
"""

from .search_space import SearchSpace, decode_solution
from .ga import run_ga
from .pso import run_pso
from .aco import run_aco
from .gsa import run_gsa
from .fuzzy_controller import FuzzyController, compute_population_diversity
from .fuzzy_ga import run_fuzzy_ga
from .fuzzy_pso import run_fuzzy_pso
from .fuzzy_aco import run_fuzzy_aco
from .hybrid_ga_pso import run_hybrid_ga_pso
from .hybrid_pso_gsa import run_hybrid_pso_gsa
from .hybrid_aco_ga import run_hybrid_aco_ga
from .novel_f_magso import run_f_magso
from .pde_robust_optimizer import run_pde_robust_opt
from .two_stage_evo import run_two_stage_evo
from .deap_ga import run_deap_ga, run_deap_ga_pinn
from .comparison import ExperimentConfig, run_experiment_grid
from .report_generator import generate_all_plots, generate_markdown_report

__all__ = [
    "SearchSpace",
    "decode_solution",
    "run_ga",
    "run_deap_ga",
    "run_deap_ga_pinn",
    "run_pso",
    "run_aco",
    "run_gsa",
    "FuzzyController",
    "compute_population_diversity",
    "run_fuzzy_ga",
    "run_fuzzy_pso",
    "run_fuzzy_aco",
    "run_hybrid_ga_pso",
    "run_hybrid_pso_gsa",
    "run_hybrid_aco_ga",
    "run_f_magso",
    "run_pde_robust_opt",
    "run_two_stage_evo",
    "ExperimentConfig",
    "run_experiment_grid",
    "generate_all_plots",
    "generate_markdown_report",
]

