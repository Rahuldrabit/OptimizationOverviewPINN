from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Burgers1DBenchmark:
    """1D Burgers' equation: u_t + u*u_x = nu*u_xx
    
    Domain: x in [x0, x1], t in [0, t1]
    Initial condition: u(x, 0) = sin(pi*x) (shock formation example)
    Boundary conditions: u(x0, t) = u(x1, t) = 0
    """
    
    x0: float = -1.0
    x1: float = 1.0
    t1: float = 1.0
    nu: float = 0.01  # viscosity parameter
    
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Returns ((x0, x1), (0, t1))"""
        return ((self.x0, self.x1), (0.0, self.t1))
    
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """u(x, 0) = sin(pi*x)"""
        return np.sin(np.pi * x)
    
    def boundary_conditions(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """u(x0, t) = 0, u(x1, t) = 0"""
        return np.zeros_like(t), np.zeros_like(t)
    
    def residual(
        self, 
        x: np.ndarray, 
        t: np.ndarray, 
        u: np.ndarray, 
        u_t: np.ndarray, 
        u_x: np.ndarray, 
        u_xx: np.ndarray
    ) -> np.ndarray:
        """PDE residual: u_t + u*u_x - nu*u_xx = 0"""
        return u_t + u * u_x - self.nu * u_xx