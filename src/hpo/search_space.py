from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchSpace:
    # Discrete architecture
    hidden_layers_min: int = 1
    hidden_layers_max: int = 6

    hidden_width_min: int = 8
    hidden_width_max: int = 256

    # Activations: match user request (tanh, sine, swish)
    activations: tuple[str, ...] = ("tanh", "sine", "swish")

    # Optimizers: Adam, AdamW, L-BFGS
    optimizers: tuple[str, ...] = ("adam", "adamw", "lbfgs")

    # Continuous ranges
    lr_min: float = 1e-4
    lr_max: float = 5e-2

    w_phys_min: float = 0.1
    w_phys_max: float = 10.0

    w_ic_min: float = 0.1
    w_ic_max: float = 50.0

    n_collocation_min: int = 64
    n_collocation_max: int = 1024


def clip_int(x: float, lo: int, hi: int) -> int:
    xi = int(round(float(x)))
    return max(lo, min(hi, xi))


def clip_float(x: float, lo: float, hi: float) -> float:
    xf = float(x)
    return max(lo, min(hi, xf))


def choose_activation(idx: float, activations: tuple[str, ...]) -> str:
    i = int(round(float(idx)))
    i = max(0, min(len(activations) - 1, i))
    return activations[i]


def choose_optimizer(idx: float, optimizers: tuple[str, ...]) -> str:
    i = int(round(float(idx)))
    i = max(0, min(len(optimizers) - 1, i))
    return optimizers[i]