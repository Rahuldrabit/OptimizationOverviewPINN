from __future__ import annotations

import unittest

import numpy as np

from src.benchmarks.burgers.burgers_1d import Burgers1DBenchmark
from src.benchmarks.heat.heat_equation import HeatEquationBenchmark
from src.benchmarks.ode.exponential_decay import ExponentialDecayBenchmark
from src.benchmarks.wave.wave_helmholtz import WaveEquationBenchmark
from src.training.benchmark_factory import get_benchmark


class TestVerifiedBenchmarks(unittest.TestCase):
    def test_ode_solution(self):
        benchmark = ExponentialDecayBenchmark()
        values = benchmark.y_true(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(values, np.exp(-np.array([0.0, 1.0, 2.0])))

    def test_heat_initial_condition(self):
        benchmark = HeatEquationBenchmark()
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(benchmark.initial_condition(x), np.sin(np.pi * x))

    def test_burgers_initial_and_boundary_conditions(self):
        benchmark = Burgers1DBenchmark()
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(benchmark.initial_condition(x), np.sin(np.pi * x))
        left, right = benchmark.boundary_conditions(np.array([0.0, 0.5]))
        np.testing.assert_allclose(left, 0.0)
        np.testing.assert_allclose(right, 0.0)

    def test_wave_initial_condition(self):
        benchmark = WaveEquationBenchmark()
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(benchmark.initial_condition_u(x), np.sin(np.pi * x))
        np.testing.assert_allclose(benchmark.initial_condition_u_t(x), 0.0)

    def test_factory_scope(self):
        for benchmark_type in ("ode", "heat", "burgers", "wave"):
            self.assertIsNotNone(get_benchmark(benchmark_type))
        with self.assertRaises(ValueError):
            get_benchmark("allen_cahn")


if __name__ == "__main__":
    unittest.main()
