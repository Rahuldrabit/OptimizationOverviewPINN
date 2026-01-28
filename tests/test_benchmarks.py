"""Test suite for all PINN benchmarks and HPO methods."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.benchmarks.ode.exponential_decay import ExponentialDecayBenchmark
from src.benchmarks.burgers.burgers_1d import Burgers1DBenchmark
from src.benchmarks.heat.heat_equation import HeatEquationBenchmark
from src.benchmarks.allen_cahn.allen_cahn import AllenCahnBenchmark
from src.benchmarks.reaction_diffusion.reaction_diffusion import ReactionDiffusionBenchmark
from src.benchmarks.navier_stokes.navier_stokes_2d import NavierStokes2DBenchmark
from src.benchmarks.wave.wave_helmholtz import WaveEquationBenchmark, HelmholtzBenchmark
from src.training.pinn_trainer import TrainConfig, train_pinn
from src.training.benchmark_factory import get_benchmark


class TestBenchmarks(unittest.TestCase):
    """Test all benchmark implementations."""

    def test_ode_benchmark(self):
        """Test ODE exponential decay benchmark."""
        bench = ExponentialDecayBenchmark()
        
        # Test domain
        domain = bench.domain()
        self.assertEqual(domain, (0.0, 5.0))
        
        # Test initial condition
        ic = bench.ic()
        self.assertEqual(ic, (0.0, 1.0))
        
        # Test analytic solution
        t = np.array([0.0, 1.0, 2.0])
        y_true = bench.y_true(t)
        expected = np.array([1.0, np.exp(-1.0), np.exp(-2.0)])
        np.testing.assert_allclose(y_true, expected, rtol=1e-10)
        
        # Test residual (should be zero for exact solution)
        y = bench.y_true(t)
        dy_dt = -bench.y_true(t)  # derivative of exp(-t) is -exp(-t)
        residual = bench.residual(t, y, dy_dt)
        np.testing.assert_allclose(residual, np.zeros_like(t), atol=1e-12)

    def test_burgers_benchmark(self):
        """Test 1D Burgers equation benchmark."""
        bench = Burgers1DBenchmark()
        
        # Test domain
        domain = bench.domain()
        self.assertEqual(domain, ((-1.0, 1.0), (0.0, 1.0)))
        
        # Test initial condition
        x = np.array([-1.0, 0.0, 1.0])
        u0 = bench.initial_condition(x)
        expected = np.sin(np.pi * x)
        np.testing.assert_allclose(u0, expected)
        
        # Test boundary conditions
        t = np.array([0.0, 0.5, 1.0])
        bc_left, bc_right = bench.boundary_conditions(t)
        np.testing.assert_allclose(bc_left, np.zeros_like(t))
        np.testing.assert_allclose(bc_right, np.zeros_like(t))

    def test_heat_equation_benchmark(self):
        """Test heat equation benchmark."""
        bench = HeatEquationBenchmark()
        
        # Test domain
        domain = bench.domain()
        self.assertEqual(domain, ((0.0, 1.0), (0.0, 1.0)))
        
        # Test initial condition
        x = np.array([0.0, 0.5, 1.0])
        u0 = bench.initial_condition(x)
        expected = np.sin(np.pi * x)
        np.testing.assert_allclose(u0, expected)
        
        # Test analytic solution at t=0 matches initial condition
        x = np.array([0.0, 0.5, 1.0])
        t = np.array([0.0, 0.0, 0.0])
        u_analytic = bench.analytic_solution(x, t)
        u_initial = bench.initial_condition(x)
        np.testing.assert_allclose(u_analytic, u_initial)

    def test_allen_cahn_benchmark(self):
        """Test Allen-Cahn equation benchmark."""
        bench = AllenCahnBenchmark()
        
        # Test domain
        domain = bench.domain()
        self.assertEqual(domain, ((-5.0, 5.0), (0.0, 2.0)))
        
        # Test initial condition (tanh profile)
        x = np.array([-1.0, 0.0, 1.0])
        u0 = bench.initial_condition(x)
        expected = np.tanh(x / np.sqrt(2 * bench.D))
        np.testing.assert_allclose(u0, expected)

    def test_reaction_diffusion_benchmark(self):
        """Test Gray-Scott reaction-diffusion benchmark."""
        bench = ReactionDiffusionBenchmark()
        
        # Test domain
        domain = bench.domain()
        self.assertEqual(domain, ((0.0, 2.5), (0.0, 2.5), (0.0, 4000.0)))
        
        # Test initial condition structure
        x = np.array([1.0, 1.25, 1.5])
        y = np.array([1.0, 1.25, 1.5])
        u0, v0 = bench.initial_condition(x, y)
        
        # Should have same shape as input
        self.assertEqual(u0.shape, x.shape)
        self.assertEqual(v0.shape, x.shape)

    def test_navier_stokes_benchmark(self):
        """Test 2D Navier-Stokes benchmark."""
        bench = NavierStokes2DBenchmark()
        
        # Test domain
        domain = bench.domain()
        self.assertEqual(domain, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
        
        # Test initial condition
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])
        u0, v0, p0 = bench.initial_condition(x, y)
        
        # Should be all zeros (fluid at rest)
        np.testing.assert_allclose(u0, np.zeros_like(x))
        np.testing.assert_allclose(v0, np.zeros_like(x))
        np.testing.assert_allclose(p0, np.zeros_like(x))
        
        # Test boundary conditions structure
        bc = bench.boundary_conditions()
        self.assertIn("top_wall", bc)
        self.assertIn("bottom_wall", bc)

    def test_wave_equation_benchmark(self):
        """Test wave equation benchmark."""
        bench = WaveEquationBenchmark()
        
        # Test domain
        domain = bench.domain()
        self.assertEqual(domain, ((0.0, 1.0), (0.0, 2.0)))
        
        # Test initial conditions
        x = np.array([0.0, 0.5, 1.0])
        u0 = bench.initial_condition_u(x)
        ut0 = bench.initial_condition_u_t(x)
        
        expected_u = np.sin(np.pi * x)
        np.testing.assert_allclose(u0, expected_u)
        np.testing.assert_allclose(ut0, np.zeros_like(x))
        
        # Test analytic solution at t=0 matches initial condition
        t = np.array([0.0, 0.0, 0.0])
        u_analytic = bench.analytic_solution(x, t)
        np.testing.assert_allclose(u_analytic, u0)

    def test_helmholtz_benchmark(self):
        """Test Helmholtz equation benchmark."""
        bench = HelmholtzBenchmark()
        
        # Test domain
        domain = bench.domain()
        self.assertEqual(domain, ((0.0, 1.0), (0.0, 1.0)))
        
        # Test analytic solution and source term consistency
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])
        
        u_analytic = bench.analytic_solution(x, y)
        f_source = bench.source_term(x, y)
        
        # Both should have same shape
        self.assertEqual(u_analytic.shape, x.shape)
        self.assertEqual(f_source.shape, x.shape)


class TestTraining(unittest.TestCase):
    """Test training functionality."""

    def test_benchmark_factory(self):
        """Test benchmark factory function."""
        # Test all benchmark types can be created
        benchmark_types = [
            "ode", "burgers", "heat", "allen_cahn",
            "reaction_diffusion", "navier_stokes", "wave", "helmholtz"
        ]
        
        for btype in benchmark_types:
            with self.subTest(benchmark_type=btype):
                bench = get_benchmark(btype)
                self.assertIsNotNone(bench)
        
        # Test invalid benchmark type
        with self.assertRaises(ValueError):
            get_benchmark("invalid")

    def test_train_config(self):
        """Test training configuration."""
        cfg = TrainConfig()
        
        # Test default values
        self.assertEqual(cfg.seed, 0)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.benchmark_type, "ode")
        self.assertEqual(cfg.hidden_layers, 3)
        self.assertEqual(cfg.hidden_width, 32)
        self.assertEqual(cfg.activation, "tanh")
        self.assertEqual(cfg.optimizer, "adam")

    def test_train_pinn_ode(self):
        """Test PINN training on ODE benchmark."""
        cfg = TrainConfig(
            seed=42,
            n_steps=100,  # Small for fast test
            n_collocation=64,
            benchmark_type="ode"
        )
        
        metrics = train_pinn(cfg)
        
        # Check that metrics are returned
        self.assertIn("config", metrics)
        self.assertIn("val_rel_l2", metrics)
        self.assertIn("val_mse", metrics)
        self.assertIn("val_linf", metrics)
        self.assertIn("train_last_loss", metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics["val_rel_l2"], 0)
        self.assertLess(metrics["val_rel_l2"], 10)  # Should be reasonable error

    def test_train_pinn_placeholder(self):
        """Test placeholder training for non-ODE benchmarks."""
        for benchmark_type in ["burgers", "heat", "allen_cahn"]:
            with self.subTest(benchmark_type=benchmark_type):
                cfg = TrainConfig(
                    benchmark_type=benchmark_type,
                    n_steps=10  # Very small for placeholder
                )
                
                metrics = train_pinn(cfg)
                
                # Should return placeholder metrics
                self.assertIn("note", metrics)
                self.assertIn("placeholder", metrics["note"].lower())


class TestHPO(unittest.TestCase):
    """Test hyperparameter optimization methods."""

    def test_search_space(self):
        """Test search space configuration."""
        from src.hpo.search_space import SearchSpace, clip_int, clip_float, choose_activation, choose_optimizer
        
        space = SearchSpace()
        
        # Test clipping functions
        self.assertEqual(clip_int(1.7, 1, 5), 2)
        self.assertEqual(clip_int(-1, 1, 5), 1)
        self.assertEqual(clip_int(10, 1, 5), 5)
        
        self.assertAlmostEqual(clip_float(0.5, 0.0, 1.0), 0.5)
        self.assertEqual(clip_float(-1.0, 0.0, 1.0), 0.0)
        
        # Test choice functions
        self.assertEqual(choose_activation(0, space.activations), "tanh")
        self.assertEqual(choose_activation(1, space.activations), "sine")
        self.assertEqual(choose_activation(2, space.activations), "swish")
        self.assertEqual(choose_activation(10, space.activations), "swish")  # clipped
        
        self.assertEqual(choose_optimizer(0, space.optimizers), "adam")
        self.assertEqual(choose_optimizer(1, space.optimizers), "adamw")
        self.assertEqual(choose_optimizer(2, space.optimizers), "lbfgs")

    def test_hpo_methods_import(self):
        """Test that HPO methods can be imported."""
        # Test imports don't fail
        from src.hpo.ga import run_ga
        from src.hpo.pso import run_pso  
        from src.hpo.aco import run_aco
        
        # Functions should be callable
        self.assertTrue(callable(run_ga))
        self.assertTrue(callable(run_pso))
        self.assertTrue(callable(run_aco))


class TestModels(unittest.TestCase):
    """Test model implementations."""

    def test_mlp_import(self):
        """Test MLP can be imported and instantiated."""
        from src.models.mlp import MLP, ModelConfig, make_model
        
        # Test MLP direct instantiation
        model = MLP(1, 1, 3, 32, "tanh")
        self.assertIsNotNone(model)
        
        # Test model factory
        cfg = ModelConfig()
        model2 = make_model(cfg)
        self.assertIsNotNone(model2)

    def test_activations(self):
        """Test activation functions."""
        from src.models.mlp import _activation, Sine
        
        # Test all supported activations can be created
        activations = ["tanh", "silu", "swish", "relu", "sine"]
        for act_name in activations:
            with self.subTest(activation=act_name):
                act = _activation(act_name)
                self.assertIsNotNone(act)
        
        # Test invalid activation
        with self.assertRaises(ValueError):
            _activation("invalid")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)