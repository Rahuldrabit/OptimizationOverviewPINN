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

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds as numpy float arrays for the 8-dim search space."""
        import numpy as np

        lb = np.array(
            [
                float(self.hidden_layers_min),
                float(self.hidden_width_min),
                0.0,
                0.0,
                float(np.log10(self.lr_min)),
                float(self.w_phys_min),
                float(self.w_ic_min),
                float(self.n_collocation_min),
            ],
            dtype=float,
        )
        ub = np.array(
            [
                float(self.hidden_layers_max),
                float(self.hidden_width_max),
                float(len(self.activations) - 1),
                float(len(self.optimizers) - 1),
                float(np.log10(self.lr_max)),
                float(self.w_phys_max),
                float(self.w_ic_max),
                float(self.n_collocation_max),
            ],
            dtype=float,
        )
        return lb, ub


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


def decode_solution(solution: Any, space: SearchSpace, base: Any) -> Any:
    """Decode an 8-element numerical solution array to a TrainConfig."""
    import numpy as np
    from dataclasses import replace

    x = np.asarray(solution, dtype=float)
    layers = clip_int(x[0], space.hidden_layers_min, space.hidden_layers_max)
    width = clip_int(x[1], space.hidden_width_min, space.hidden_width_max)
    activation = choose_activation(x[2], space.activations)
    optimizer = choose_optimizer(x[3], space.optimizers)

    log10_lr = clip_float(x[4], float(np.log10(space.lr_min)), float(np.log10(space.lr_max)))
    lr = float(10 ** log10_lr)

    w_phys = clip_float(x[5], space.w_phys_min, space.w_phys_max)
    w_ic = clip_float(x[6], space.w_ic_min, space.w_ic_max)
    n_col = clip_int(x[7], space.n_collocation_min, space.n_collocation_max)

    return replace(
        base,
        hidden_layers=layers,
        hidden_width=width,
        activation=activation,
        optimizer=optimizer,
        lr=lr,
        w_phys=w_phys,
        w_ic=w_ic,
        n_collocation=n_col,
    )