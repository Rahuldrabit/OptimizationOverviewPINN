from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeatEquationBenchmark:
    """1D Heat equation: u_t = alpha * u_xx
    
    Domain: x in [x0, x1], t in [0, t1]
    Initial condition: u(x, 0) = sin(pi*x/(x1-x0))
    Boundary conditions: u(x0, t) = u(x1, t) = 0
    Analytic solution: u(x,t) = sin(pi*x/(x1-x0)) * exp(-alpha*pi^2*t/(x1-x0)^2)
    """
    
    x0: float = 0.0
    x1: float = 1.0
    t1: float = 1.0
    alpha: float = 0.1  # thermal diffusivity
    
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Returns ((x0, x1), (0, t1))"""
        return ((self.x0, self.x1), (0.0, self.t1))
    
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """u(x, 0) = sin(pi*x/(x1-x0))"""
        L = self.x1 - self.x0
        return np.sin(np.pi * (x - self.x0) / L)
    
    def boundary_conditions(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """u(x0, t) = 0, u(x1, t) = 0"""
        return np.zeros_like(t), np.zeros_like(t)
    
    def analytic_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Exact solution: u(x,t) = sin(pi*x/L) * exp(-alpha*pi^2*t/L^2)"""
        L = self.x1 - self.x0
        return (
            np.sin(np.pi * (x - self.x0) / L) * 
            np.exp(-self.alpha * np.pi**2 * t / L**2)
        )
    
    def residual(
        self, 
        x: np.ndarray, 
        t: np.ndarray, 
        u: np.ndarray, 
        u_t: np.ndarray, 
        u_xx: np.ndarray
    ) -> np.ndarray:
        """PDE residual: u_t - alpha * u_xx = 0"""
        return u_t - self.alpha * u_xx