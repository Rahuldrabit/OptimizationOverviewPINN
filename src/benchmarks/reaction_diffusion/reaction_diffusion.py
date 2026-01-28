from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReactionDiffusionBenchmark:
    """Gray-Scott reaction-diffusion system:
    u_t = D_u * (u_xx + u_yy) - u*v^2 + f*(1-u)
    v_t = D_v * (v_xx + v_yy) + u*v^2 - (f+k)*v
    
    Domain: x,y in [0, L] x [0, L], t in [0, t1]
    """
    
    L: float = 2.5  # domain size
    t1: float = 4000.0
    D_u: float = 2e-5  # diffusion coefficient for u
    D_v: float = 1e-5  # diffusion coefficient for v
    f: float = 0.054   # feed rate
    k: float = 0.063   # kill rate
    
    def domain(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Returns ((0, L), (0, L), (0, t1))"""
        return ((0.0, self.L), (0.0, self.L), (0.0, self.t1))
    
    def initial_condition(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Initial conditions with small random perturbation around center"""
        u = np.ones_like(x)
        v = np.zeros_like(x)
        
        # Add small perturbation in center region
        center_x, center_y = self.L / 2, self.L / 2
        radius = 0.1
        mask = (x - center_x)**2 + (y - center_y)**2 < radius**2
        
        u[mask] = 0.5 + 0.1 * np.random.random(np.sum(mask))
        v[mask] = 0.25 + 0.1 * np.random.random(np.sum(mask))
        
        return u, v
    
    def boundary_conditions(self) -> str:
        """Periodic boundary conditions"""
        return "periodic"
    
    def residual_u(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        t: np.ndarray,
        u: np.ndarray, 
        v: np.ndarray,
        u_t: np.ndarray, 
        u_xx: np.ndarray,
        u_yy: np.ndarray
    ) -> np.ndarray:
        """PDE residual for u: u_t - D_u*(u_xx + u_yy) + u*v^2 - f*(1-u) = 0"""
        return u_t - self.D_u * (u_xx + u_yy) + u * v**2 - self.f * (1 - u)
    
    def residual_v(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        t: np.ndarray,
        u: np.ndarray, 
        v: np.ndarray,
        v_t: np.ndarray, 
        v_xx: np.ndarray,
        v_yy: np.ndarray
    ) -> np.ndarray:
        """PDE residual for v: v_t - D_v*(v_xx + v_yy) - u*v^2 + (f+k)*v = 0"""
        return v_t - self.D_v * (v_xx + v_yy) - u * v**2 + (self.f + self.k) * v