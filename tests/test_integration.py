from __future__ import annotations

import unittest

from src.hpo.pde_robust_optimizer import run_pde_robust_opt
from src.training.pinn_trainer import TrainConfig, train_pinn


class TestPDERobustWorkflow(unittest.TestCase):
    def test_training_workflow(self):
        metrics = train_pinn(TrainConfig(benchmark_type="ode", n_steps=5, n_collocation=32))
        self.assertIn("val_rel_l2", metrics)
        self.assertGreater(metrics["val_rel_l2"], 0.0)

    def test_reproducibility(self):
        config = TrainConfig(seed=42, n_steps=5, n_collocation=32)
        first = train_pinn(config)
        second = train_pinn(config)
        self.assertEqual(first["val_rel_l2"], second["val_rel_l2"])

    def test_optimizer_smoke(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        output_dir = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
        metrics = run_pde_robust_opt(
            str(output_dir),
            benchmark_type="ode",
            seed=0,
            n_generations=1,
            sol_per_pop=4,
            n_steps=2,
        )
        self.assertIn("val_rel_l2", metrics)
        self.assertEqual(metrics["n_evaluations"], 8)
        self.assertTrue((output_dir / "pde_robust_de_best_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
