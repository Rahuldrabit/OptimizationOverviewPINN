from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllenCahnBenchmark:
    """Allen-Cahn equation: u_t = D * u_xx + u - u^3
    
    Domain: x in [x0, x1], t in [0, t1]
    Initial condition: u(x, 0) = tanh((x-x_center)/sqrt(2*D))
    Boundary conditions: Neumann (u_x = 0) at boundaries
    """
    
    x0: float = -5.0
    x1: float = 5.0
    t1: float = 2.0
    D: float = 0.01  # diffusion coefficient
    x_center: float = 0.0  # initial interface position
    
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Returns ((x0, x1), (0, t1))"""
        return ((self.x0, self.x1), (0.0, self.t1))
    
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """u(x, 0) = tanh((x-x_center)/sqrt(2*D))"""
        return np.tanh((x - self.x_center) / np.sqrt(2 * self.D))
    
    def boundary_conditions_neumann(self) -> tuple[float, float]:
        """Neumann BC: u_x(x0, t) = u_x(x1, t) = 0"""
        return 0.0, 0.0
    
    def residual(
        self, 
        x: np.ndarray, 
        t: np.ndarray, 
        u: np.ndarray, 
        u_t: np.ndarray, 
        u_xx: np.ndarray
    ) -> np.ndarray:
        """PDE residual: u_t - D * u_xx - u + u^3 = 0"""
        return u_t - self.D * u_xx - u + u**3