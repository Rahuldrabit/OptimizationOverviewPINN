# PDE-Robust-DE for PINNs

This repository contains a focused implementation of PDE-Robust Differential Evolution (PDE-Robust-DE) for hyperparameter optimization of Physics-Informed Neural Networks.

## Supported Benchmarks

The runtime supports four verified trainers:

- `ode`
- `heat`
- `burgers`
- `wave`

Unsupported benchmark names fail immediately. No placeholder metrics are used.

## Run PDE-Robust-DE

Single benchmark and seed:

```powershell
python scripts\run_pde_robust_de.py ode --seed 0 --generations 10 --pop-size 20 --steps 1200
```

Ten-seed confirmation batch:

```powershell
python scripts\run_pde_robust_de.py heat --seeds 0 1 2 3 4 5 6 7 8 9 --generations 10 --pop-size 20 --steps 1200
```

Use `--quick` for a short smoke test. Results are written to `outputs/pde_robust_de/<benchmark>/`, including one JSON file per run and `pde_robust_de_summary.json` for a seed batch.

## Project Layout

```text
src/benchmarks/       Verified ODE, Heat, Burgers, and Wave benchmarks
src/models/           PINN MLP model
src/training/         PINN training and benchmark factory
src/hpo/              PDE-Robust-DE and its search-space utilities
scripts/              PDE-Robust-DE runner and test runner
paper/                Manuscript and LaTeX tables
outputs/              Local experiment results, ignored by Git
```

## Tests

```powershell
pytest -q
```

PyTorch is required for genuine PINN training. The project no longer falls back to synthetic or placeholder metrics.
