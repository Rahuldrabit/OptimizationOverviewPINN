from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


if HAS_TORCH:
    class Sine(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.sin(x)


    def _activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "tanh":
            return nn.Tanh()
        if name in {"silu", "swish"}:
            # PyTorch's SiLU is equivalent to Swish.
            return nn.SiLU()
        if name == "relu":
            return nn.ReLU()
        if name == "sine":
            return Sine()
        raise ValueError(f"Unknown activation: {name}")


    class MLP(nn.Module):
        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_layers: int,
            hidden_width: int,
            activation: str = "tanh",
        ) -> None:
            super().__init__()
            act = _activation(activation)

            layers: list[nn.Module] = []
            last = in_dim
            for _ in range(int(hidden_layers)):
                layers.append(nn.Linear(last, int(hidden_width)))
                layers.append(act)
                last = int(hidden_width)
            layers.append(nn.Linear(last, out_dim))

            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:
    class Sine:
        def __call__(self, x):
            import numpy as np
            return np.sin(x)


    def _activation(name: str):
        import numpy as np
        name = name.lower()
        if name == "tanh":
            return np.tanh
        if name in {"silu", "swish"}:
            return lambda x: x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
        if name == "relu":
            return lambda x: np.maximum(0.0, x)
        if name == "sine":
            return Sine()
        raise ValueError(f"Unknown activation: {name}")


    class MLP:
        """Pure NumPy fallback for MLP when PyTorch is not installed."""
        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_layers: int,
            hidden_width: int,
            activation: str = "tanh",
        ) -> None:
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.hidden_layers = int(hidden_layers)
            self.hidden_width = int(hidden_width)
            self.activation_name = activation
            self.act = _activation(activation)

        def __call__(self, x):
            import numpy as np
            return np.zeros((len(x), self.out_dim), dtype=np.float32)


@dataclass(frozen=True)
class ModelConfig:
    hidden_layers: int = 3
    hidden_width: int = 32
    activation: str = "tanh"


def make_model(cfg: ModelConfig):
    return MLP(1, 1, cfg.hidden_layers, cfg.hidden_width, cfg.activation)