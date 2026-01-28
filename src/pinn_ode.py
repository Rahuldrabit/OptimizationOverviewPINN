from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .ode_benchmarks import ExponentialDecayBenchmark
from .utils import set_seed, try_set_torch_seed


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    device: str = "cpu"  # "cpu" or "cuda"

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


def _to_device(device: str):
    import torch

    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_collocation(n: int, t0: float, t1: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.uniform(t0, t1, size=(int(n), 1))
    return t.astype(np.float32)


def train_pinn(cfg: TrainConfig) -> dict[str, Any]:
    """Train PINN on y' = -y, y(0)=1.

    Returns a dict with metrics suitable for HPO.
    """

    set_seed(cfg.seed)
    try_set_torch_seed(cfg.seed)

    import torch
    bench = ExponentialDecayBenchmark(t0=cfg.t0, t1=cfg.t1, y0=1.0)
    device = _to_device(cfg.device)

    from .models import MLP

    model = MLP(1, 1, cfg.hidden_layers, cfg.hidden_width, cfg.activation).to(device)

    opt_name = cfg.optimizer.lower()
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
        use_lbfgs = False
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))
        use_lbfgs = False
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=float(cfg.lr),
            max_iter=int(cfg.lbfgs_max_iter),
        )
        use_lbfgs = True
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    t_col_np = _make_collocation(cfg.n_collocation, cfg.t0, cfg.t1, seed=cfg.seed + 123)
    t_col = torch.tensor(t_col_np, device=device, requires_grad=True)

    t0_tensor = torch.tensor([[cfg.t0]], device=device, requires_grad=True)
    y0_target = torch.tensor([[1.0]], device=device)

    last_loss = None

    if use_lbfgs:
        # L-BFGS uses a closure-based interface.
        for _ in range(int(cfg.n_steps)):
            def closure():
                opt.zero_grad()

                y = model(t_col)
                dy_dt = torch.autograd.grad(
                    outputs=y,
                    inputs=t_col,
                    grad_outputs=torch.ones_like(y),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                r = dy_dt + y
                loss_phys = torch.mean(r**2)

                y0_pred = model(t0_tensor)
                loss_ic = torch.mean((y0_pred - y0_target) ** 2)

                loss = float(cfg.w_phys) * loss_phys + float(cfg.w_ic) * loss_ic
                loss.backward()
                return loss

            loss_tensor = opt.step(closure)
            last_loss = float(loss_tensor.detach().cpu().item())
    else:
        for _ in range(int(cfg.n_steps)):
            opt.zero_grad(set_to_none=True)

            # Physics residual: dy/dt + y = 0
            y = model(t_col)
            dy_dt = torch.autograd.grad(
                outputs=y,
                inputs=t_col,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True,
            )[0]
            r = dy_dt + y
            loss_phys = torch.mean(r**2)

            # Initial condition
            y0_pred = model(t0_tensor)
            loss_ic = torch.mean((y0_pred - y0_target) ** 2)

            loss = float(cfg.w_phys) * loss_phys + float(cfg.w_ic) * loss_ic
            loss.backward()
            opt.step()

            last_loss = float(loss.detach().cpu().item())

    # Evaluation on grid
    t_eval = np.linspace(cfg.t0, cfg.t1, int(cfg.n_eval), dtype=np.float32).reshape(-1, 1)
    y_true = bench.y_true(t_eval).reshape(-1, 1).astype(np.float32)

    with torch.no_grad():
        t_eval_t = torch.tensor(t_eval, device=device)
        y_pred = model(t_eval_t).detach().cpu().numpy()

    err = y_pred - y_true
    mse = float(np.mean(err**2))
    linf = float(np.max(np.abs(err)))
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(y_true) + 1e-12))

    return {
        "config": asdict(cfg),
        "train_last_loss": last_loss,
        "val_mse": mse,
        "val_linf": linf,
        "val_rel_l2": rel_l2,
    }
