"""Integration tests for full PINN training workflows."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.pinn_trainer import TrainConfig, train_pinn


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""

    def test_full_training_workflow(self):
        """Test complete training workflow with different configurations."""
        
        configs_to_test = [
            # Basic ODE config
            {"benchmark_type": "ode", "n_steps": 100, "activation": "tanh"},
            # Different activations
            {"benchmark_type": "ode", "n_steps": 100, "activation": "sine"},
            {"benchmark_type": "ode", "n_steps": 100, "activation": "swish"},
            # Different optimizers
            {"benchmark_type": "ode", "n_steps": 100, "optimizer": "adam"},
            {"benchmark_type": "ode", "n_steps": 100, "optimizer": "adamw"},
            {"benchmark_type": "ode", "n_steps": 50, "optimizer": "lbfgs"},  # Fewer steps for LBFGS
        ]
        
        for i, config_dict in enumerate(configs_to_test):
            with self.subTest(config=i):
                cfg = TrainConfig(**config_dict)
                metrics = train_pinn(cfg)
                
                # Basic sanity checks
                self.assertIn("val_rel_l2", metrics)
                self.assertGreater(metrics["val_rel_l2"], 0)
                
                # Config should be preserved
                self.assertEqual(metrics["config"]["benchmark_type"], cfg.benchmark_type)
                self.assertEqual(metrics["config"]["activation"], cfg.activation)

    def test_reproducibility(self):
        """Test that training is reproducible with same seed."""
        cfg = TrainConfig(seed=42, n_steps=50)
        
        # Run twice with same config
        metrics1 = train_pinn(cfg)
        metrics2 = train_pinn(cfg)
        
        # Should get identical results
        self.assertEqual(metrics1["val_rel_l2"], metrics2["val_rel_l2"])
        self.assertEqual(metrics1["val_mse"], metrics2["val_mse"])

    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        cfg1 = TrainConfig(seed=42, n_steps=50)
        cfg2 = TrainConfig(seed=123, n_steps=50)
        
        metrics1 = train_pinn(cfg1)
        metrics2 = train_pinn(cfg2)
        
        # Should get different results (with high probability)
        self.assertNotEqual(metrics1["val_rel_l2"], metrics2["val_rel_l2"])

    def test_placeholder_benchmarks(self):
        """Test that placeholder benchmarks work for all PDE types."""
        pde_types = ["burgers", "heat", "allen_cahn", "reaction_diffusion", "navier_stokes", "wave", "helmholtz"]
        
        for pde_type in pde_types:
            with self.subTest(benchmark=pde_type):
                cfg = TrainConfig(benchmark_type=pde_type, n_steps=10)
                metrics = train_pinn(cfg)
                
                # Should return placeholder metrics
                self.assertIn("note", metrics)
                self.assertIn("placeholder", metrics["note"].lower())
                self.assertEqual(metrics["config"]["benchmark_type"], pde_type)


if __name__ == "__main__":
    unittest.main(verbosity=2)