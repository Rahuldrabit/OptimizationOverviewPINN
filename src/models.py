from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


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


@dataclass(frozen=True)
class ModelConfig:
    hidden_layers: int = 3
    hidden_width: int = 32
    activation: str = "tanh"


def make_model(cfg: ModelConfig) -> nn.Module:
    return MLP(1, 1, cfg.hidden_layers, cfg.hidden_width, cfg.activation)
