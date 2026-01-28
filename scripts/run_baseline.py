from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.pinn_trainer import TrainConfig, train_pinn
from utils import ensure_dir, save_json


def main() -> None:
    """Run baseline PINN training on specified benchmark."""
    benchmark_type = sys.argv[1] if len(sys.argv) > 1 else "ode"
    
    out_dir = os.path.join("outputs", "baseline", benchmark_type)
    ensure_dir(out_dir)

    cfg = TrainConfig(
        seed=0,
        benchmark_type=benchmark_type,
        n_steps=2000,
        n_collocation=256,
        hidden_layers=3,
        hidden_width=32,
        activation="tanh",
        optimizer="adam",
        lr=1e-3,
        w_phys=1.0,
        w_ic=10.0,
    )
    
    print(f"Training PINN on {benchmark_type} benchmark...")
    metrics = train_pinn(cfg)
    save_json(os.path.join(out_dir, "metrics.json"), metrics)
    
    print(f"\nResults for {benchmark_type}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
