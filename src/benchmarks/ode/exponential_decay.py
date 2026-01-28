from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExponentialDecayBenchmark:
    """ODE: y'(t) = -y(t), y(t0)=y0. Analytic: y(t)=y0*exp(-(t-t0))."""

    t0: float = 0.0
    t1: float = 5.0
    y0: float = 1.0

    def domain(self) -> tuple[float, float]:
        return (self.t0, self.t1)

    def ic(self) -> tuple[float, float]:
        return (self.t0, self.y0)

    def y_true(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.y0 * np.exp(-(t - self.t0))

    def residual(self, t: np.ndarray, y: np.ndarray, dy_dt: np.ndarray) -> np.ndarray:
        # Residual for y' + y = 0
        return dy_dt + y