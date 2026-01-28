"""Run all benchmarks with baseline configuration for comparison."""

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
    """Run baseline PINN on all available benchmarks."""
    
    benchmarks = [
        "ode",
        "burgers", 
        "heat",
        "allen_cahn",
        "reaction_diffusion",
        "navier_stokes",
        "wave",
        "helmholtz"
    ]
    
    results = {}
    
    for benchmark in benchmarks:
        print(f"\n{'='*50}")
        print(f"Training {benchmark.upper()} benchmark")
        print(f"{'='*50}")
        
        out_dir = os.path.join("outputs", "baseline", benchmark)
        ensure_dir(out_dir)

        cfg = TrainConfig(
            seed=42,  # Fixed seed for reproducible comparison
            benchmark_type=benchmark,
            n_steps=1000,  # Moderate steps for comparison
            n_collocation=256,
            hidden_layers=3,
            hidden_width=64,
            activation="tanh",
            optimizer="adam",
            lr=1e-3,
            w_phys=1.0,
            w_ic=10.0,
        )
        
        try:
            metrics = train_pinn(cfg)
            save_json(os.path.join(out_dir, "metrics.json"), metrics)
            
            # Store key metrics for comparison
            results[benchmark] = {
                "val_rel_l2": metrics["val_rel_l2"],
                "val_mse": metrics["val_mse"],
                "train_last_loss": metrics["train_last_loss"],
                "config": {
                    "activation": cfg.activation,
                    "optimizer": cfg.optimizer,
                    "lr": cfg.lr,
                    "hidden_layers": cfg.hidden_layers,
                    "hidden_width": cfg.hidden_width
                }
            }
            
            print(f"✓ {benchmark}: val_rel_l2 = {metrics['val_rel_l2']:.6f}")
            
        except Exception as e:
            print(f"✗ {benchmark}: Failed - {str(e)}")
            results[benchmark] = {"error": str(e)}
    
    # Save comparison results
    comparison_dir = os.path.join("outputs", "comparison")
    ensure_dir(comparison_dir)
    save_json(os.path.join(comparison_dir, "baseline_comparison.json"), results)
    
    print(f"\n{'='*50}")
    print("BENCHMARK COMPARISON SUMMARY")
    print(f"{'='*50}")
    
    for benchmark, result in results.items():
        if "error" in result:
            print(f"{benchmark:20s}: ERROR - {result['error']}")
        else:
            print(f"{benchmark:20s}: val_rel_l2 = {result['val_rel_l2']:.6f}")
    
    print(f"\nDetailed results saved to: outputs/comparison/baseline_comparison.json")


if __name__ == "__main__":
    main()