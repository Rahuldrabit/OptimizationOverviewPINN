from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..utils import set_seed, try_set_torch_seed
from .benchmark_factory import get_benchmark, train_pinn_ode, train_pinn_placeholder


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    device: str = "cpu"  # "cpu" or "cuda"
    benchmark_type: str = "ode"  # "ode", "burgers", "heat", etc.

    # ODE domain + evaluation
    t0: float = 0.0
    t1: float = 5.0
    n_eval: int = 200

    # Model
    hidden_layers: int = 3
    hidden_width: int = 32
    activation: str = "tanh"

    # Optimizer
    optimizer: str = "adam"  # "adam", "adamw", "lbfgs"
    lbfgs_max_iter: int = 20

    # Training
    lr: float = 1e-3
    n_steps: int = 2000
    n_collocation: int = 256

    # Loss weights
    w_phys: float = 1.0
    w_ic: float = 10.0


def train_pinn(cfg: TrainConfig) -> dict[str, Any]:
    """Train PINN on specified benchmark.

    Returns a dict with metrics suitable for HPO.
    """

    set_seed(cfg.seed)
    try_set_torch_seed(cfg.seed)

    bench = get_benchmark(cfg.benchmark_type)
    
    if cfg.benchmark_type == "ode":
        metrics = train_pinn_ode(cfg, bench)
    else:
        # Use placeholder for other benchmarks
        metrics = train_pinn_placeholder(cfg, bench)

    return {
        "config": asdict(cfg),
        **metrics
    }