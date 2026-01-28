from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaveEquationBenchmark:
    """1D Wave equation: u_tt = c^2 * u_xx
    
    Domain: x in [x0, x1], t in [0, t1]
    Initial conditions: u(x, 0) = sin(pi*x/L), u_t(x, 0) = 0
    Boundary conditions: u(x0, t) = u(x1, t) = 0
    Analytic solution: u(x,t) = sin(pi*x/L) * cos(c*pi*t/L)
    """
    
    x0: float = 0.0
    x1: float = 1.0
    t1: float = 2.0
    c: float = 1.0  # wave speed
    
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Returns ((x0, x1), (0, t1))"""
        return ((self.x0, self.x1), (0.0, self.t1))
    
    def initial_condition_u(self, x: np.ndarray) -> np.ndarray:
        """u(x, 0) = sin(pi*x/L)"""
        L = self.x1 - self.x0
        return np.sin(np.pi * (x - self.x0) / L)
    
    def initial_condition_u_t(self, x: np.ndarray) -> np.ndarray:
        """u_t(x, 0) = 0"""
        return np.zeros_like(x)
    
    def boundary_conditions(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """u(x0, t) = 0, u(x1, t) = 0"""
        return np.zeros_like(t), np.zeros_like(t)
    
    def analytic_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Exact solution: u(x,t) = sin(pi*x/L) * cos(c*pi*t/L)"""
        L = self.x1 - self.x0
        return (
            np.sin(np.pi * (x - self.x0) / L) * 
            np.cos(self.c * np.pi * t / L)
        )
    
    def residual(
        self, 
        x: np.ndarray, 
        t: np.ndarray, 
        u: np.ndarray, 
        u_tt: np.ndarray, 
        u_xx: np.ndarray
    ) -> np.ndarray:
        """PDE residual: u_tt - c^2 * u_xx = 0"""
        return u_tt - self.c**2 * u_xx


@dataclass(frozen=True)
class HelmholtzBenchmark:
    """Helmholtz equation: u_xx + u_yy + k^2 * u = f(x,y)
    
    Domain: x,y in [0, L] x [0, L]
    Source term: f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)
    Boundary conditions: u = 0 on boundary
    Analytic solution: u(x,y) = sin(pi*x)*sin(pi*y) when k^2 = 2*pi^2
    """
    
    L: float = 1.0  # domain size
    k: float = 2 * np.pi  # wave number (k^2 = 2*pi^2 for exact solution)
    
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Returns ((0, L), (0, L))"""
        return ((0.0, self.L), (0.0, self.L))
    
    def source_term(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)"""
        return 2 * np.pi**2 * np.sin(np.pi * x / self.L) * np.sin(np.pi * y / self.L)
    
    def boundary_conditions(self) -> dict[str, float]:
        """u = 0 on all boundaries"""
        return {"all_boundaries": 0.0}
    
    def analytic_solution(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact solution: u(x,y) = sin(pi*x/L)*sin(pi*y/L)"""
        return np.sin(np.pi * x / self.L) * np.sin(np.pi * y / self.L)
    
    def residual(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        u: np.ndarray, 
        u_xx: np.ndarray, 
        u_yy: np.ndarray
    ) -> np.ndarray:
        """PDE residual: u_xx + u_yy + k^2*u - f(x,y) = 0"""
        f = self.source_term(x, y)
        return u_xx + u_yy + self.k**2 * u - f