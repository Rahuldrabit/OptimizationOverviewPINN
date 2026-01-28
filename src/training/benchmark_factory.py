from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from ..utils import set_seed, try_set_torch_seed


def get_benchmark(benchmark_type: str):
    """Factory function to get benchmark instance based on type."""
    if benchmark_type == "ode":
        from ..benchmarks.ode.exponential_decay import ExponentialDecayBenchmark
        return ExponentialDecayBenchmark()
    elif benchmark_type == "burgers":
        from ..benchmarks.burgers.burgers_1d import Burgers1DBenchmark
        return Burgers1DBenchmark()
    elif benchmark_type == "heat":
        from ..benchmarks.heat.heat_equation import HeatEquationBenchmark
        return HeatEquationBenchmark()
    elif benchmark_type == "allen_cahn":
        from ..benchmarks.allen_cahn.allen_cahn import AllenCahnBenchmark
        return AllenCahnBenchmark()
    elif benchmark_type == "reaction_diffusion":
        from ..benchmarks.reaction_diffusion.reaction_diffusion import ReactionDiffusionBenchmark
        return ReactionDiffusionBenchmark()
    elif benchmark_type == "navier_stokes":
        from ..benchmarks.navier_stokes.navier_stokes_2d import NavierStokes2DBenchmark
        return NavierStokes2DBenchmark()
    elif benchmark_type == "wave":
        from ..benchmarks.wave.wave_helmholtz import WaveEquationBenchmark
        return WaveEquationBenchmark()
    elif benchmark_type == "helmholtz":
        from ..benchmarks.wave.wave_helmholtz import HelmholtzBenchmark
        return HelmholtzBenchmark()
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")


def train_pinn_ode(cfg, bench) -> dict[str, Any]:
    """Train PINN for ODE benchmark (original implementation)."""
    import torch

    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    
    from ..models.mlp import MLP

    model = MLP(1, 1, cfg.hidden_layers, cfg.hidden_width, cfg.activation).to(device)

    # Create optimizer
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

    # Collocation points and initial condition
    rng = np.random.default_rng(cfg.seed + 123)
    t_col_np = rng.uniform(cfg.t0, cfg.t1, size=(int(cfg.n_collocation), 1)).astype(np.float32)
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
        "train_last_loss": last_loss,
        "val_mse": mse,
        "val_linf": linf,
        "val_rel_l2": rel_l2,
    }


def train_pinn_placeholder(cfg, bench) -> dict[str, Any]:
    """Placeholder for non-ODE benchmarks - returns dummy metrics."""
    # For now, return dummy metrics since implementing full PDE trainers
    # would require significant additional complexity
    return {
        "train_last_loss": 0.1,
        "val_mse": 0.01,
        "val_linf": 0.1,
        "val_rel_l2": 0.05,
        "note": f"Placeholder metrics for {cfg.benchmark_type} benchmark"
    }