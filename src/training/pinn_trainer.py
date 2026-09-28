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
    )
except (ImportError, ValueError):
    from utils import set_seed, try_set_torch_seed
    from training.benchmark_factory import (
        get_benchmark,
        train_pinn_ode,
        train_pinn_heat,
        train_pinn_burgers,
        train_pinn_wave,
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
        raise ValueError(
            f"Unsupported benchmark '{cfg.benchmark_type}'. "
            "Supported benchmarks: ode, heat, burgers, wave."
        )

    import math
    for k in ["val_rel_l2", "val_mse", "val_linf", "train_last_loss"]:
        v = metrics.get(k)
        if v is None or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            metrics[k] = 1e6  # Large penalty for divergence or missing metric

    return {
        "config": asdict(cfg),
        **metrics
    }