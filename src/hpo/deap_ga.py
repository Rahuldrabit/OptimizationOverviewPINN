"""DEAP-based Genetic Algorithm Implementation for PINN Hyperparameter Optimization.

Serves as the external ground-truth validation framework (Distributed Evolutionary
Algorithms in Python - Université Laval) to cross-validate our metaheuristic implementations.
"""

from __future__ import annotations

import random
from typing import Any, Callable
import numpy as np

from deap import base, creator, tools, algorithms

try:
    from ..training.pinn_trainer import TrainConfig, train_pinn
    from ..utils import ensure_dir, save_json
    from .search_space import SearchSpace, decode_solution
except (ImportError, ValueError):
    from training.pinn_trainer import TrainConfig, train_pinn
    from utils import ensure_dir, save_json
    from hpo.search_space import SearchSpace, decode_solution


def _setup_deap_environment():
    """Safely initialize DEAP creator types if not already registered."""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)


def run_deap_ga(
    lb: np.ndarray,
    ub: np.ndarray,
    eval_func: Callable[[np.ndarray], float],
    pop_size: int = 15,
    n_generations: int = 10,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, float, list[float]]:
    """Execute DEAP standard Genetic Algorithm (eaSimple) under bounded search space."""
    _setup_deap_environment()
    random.seed(seed)
    np.random.seed(seed)

    dim = len(lb)
    toolbox = base.Toolbox()

    # Attribute generator for bounded continuous/discrete genes
    def create_individual():
        genes = [random.uniform(lb[i], ub[i]) for i in range(dim)]
        return creator.Individual(genes)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function wrapper
    def evaluate(individual):
        arr = np.array(individual)
        # Apply boundary clipping
        arr = np.clip(arr, lb, ub)
        val = eval_func(arr)
        return (val,)

    toolbox.register("evaluate", evaluate)

    # Genetic Operators: Two-point crossover, Gaussian mutation, Tournament selection
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    sigmas = [0.1 * (ub[i] - lb[i]) for i in range(dim)]

    def mutate_bounded(individual):
        tools.mutGaussian(individual, mu=0, sigma=sigmas, indpb=0.2)
        for i in range(dim):
            individual[i] = max(lb[i], min(ub[i], individual[i]))
        return individual,

    toolbox.register("mutate", mutate_bounded)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    # Run standard generational algorithm
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations,
        stats=stats, halloffame=hof, verbose=False
    )

    best_individual = np.clip(np.array(hof[0]), lb, ub)
    best_fitness = float(hof[0].fitness.values[0])
    history = [float(record["min"]) for record in logbook]

    return best_individual, best_fitness, history


def run_deap_ga_pinn(
    out_dir: str = "outputs/deap_validation",
    benchmark_type: str = "ode",
    seed: int = 0,
    pop_size: int = 15,
    n_generations: int = 10,
    n_steps: int = 1200,
) -> dict[str, Any]:
    """Execute DEAP GA on the target PINN benchmark."""
    space = SearchSpace()
    base_cfg = TrainConfig(seed=seed, n_steps=n_steps, benchmark_type=benchmark_type)
    lb, ub = space.get_bounds()

    def eval_fn(cand: np.ndarray) -> float:
        cfg = decode_solution(cand, space, base_cfg)
        metrics = train_pinn(cfg)
        return float(metrics["val_rel_l2"])

    best_x, best_fit, history = run_deap_ga(
        lb, ub, eval_fn, pop_size=pop_size, n_generations=n_generations, seed=seed
    )

    best_cfg = decode_solution(best_x, space, base_cfg)
    best_metrics = train_pinn(best_cfg)
    best_metrics["history"] = history
    best_metrics["optimizer_name"] = "DEAP-GA (Standard Baseline)"

    ensure_dir(out_dir)
    save_json(f"{out_dir}/deap_ga_best_metrics.json", best_metrics)
    return best_metrics
