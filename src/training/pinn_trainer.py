from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    from ..utils import set_seed, try_set_torch_seed
    from .benchmark_factory import (
        get_benchmark,
        train_pinn_ode,
        train_pinn_heat,
        train_pinn_burgers,
        train_pinn_wave,
        train_pinn_placeholder,
    )
except (ImportError, ValueError):
    from utils import set_seed, try_set_torch_seed
    from training.benchmark_factory import (
        get_benchmark,
        train_pinn_ode,
        train_pinn_heat,
        train_pinn_burgers,
        train_pinn_wave,
        train_pinn_placeholder,
    )



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
    lbfgs_max_iter: int = 3  # reduced from 20: keeps L-BFGS candidates from dominating HPO wall-clock budget

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
    elif cfg.benchmark_type == "heat":
        metrics = train_pinn_heat(cfg, bench)
    elif cfg.benchmark_type == "burgers":
        metrics = train_pinn_burgers(cfg, bench)
    elif cfg.benchmark_type == "wave":
        metrics = train_pinn_wave(cfg, bench)
    else:
        # Real trainer not yet implemented for this PDE (allen_cahn, reaction_diffusion,
        # navier_stokes, helmholtz) - falls back to fixed placeholder metrics. Any results
        # for these benchmark types are NOT real and must not be reported as such.
        metrics = train_pinn_placeholder(cfg, bench)

    return {
        "config": asdict(cfg),
        **metrics
    }