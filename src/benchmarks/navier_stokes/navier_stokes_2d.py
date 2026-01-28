from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NavierStokes2DBenchmark:
    """2D Navier-Stokes equations (incompressible):
    u_t + u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)
    v_t + u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)
    u_x + v_y = 0  (continuity equation)
    
    Domain: x,y in [0, L] x [0, L], t in [0, t1]
    Lid-driven cavity flow benchmark
    """
    
    L: float = 1.0  # domain size
    t1: float = 1.0
    nu: float = 0.01  # kinematic viscosity
    u_lid: float = 1.0  # lid velocity
    
    def domain(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Returns ((0, L), (0, L), (0, t1))"""
        return ((0.0, self.L), (0.0, self.L), (0.0, self.t1))
    
    def initial_condition(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """u(x,y,0) = v(x,y,0) = 0, p(x,y,0) = 0 (fluid at rest)"""
        u = np.zeros_like(x)
        v = np.zeros_like(x)
        p = np.zeros_like(x)
        return u, v, p
    
    def boundary_conditions(self) -> dict[str, tuple[str, float]]:
        """Boundary conditions for lid-driven cavity:
        - Top wall (y=L): u=u_lid, v=0 (moving lid)
        - Other walls: u=v=0 (no-slip)
        - Pressure: one point fixed to remove indeterminacy
        """
        return {
            "top_wall": ("dirichlet", {"u": self.u_lid, "v": 0.0}),
            "bottom_wall": ("dirichlet", {"u": 0.0, "v": 0.0}),
            "left_wall": ("dirichlet", {"u": 0.0, "v": 0.0}),
            "right_wall": ("dirichlet", {"u": 0.0, "v": 0.0}),
            "pressure_ref": ("dirichlet", {"p": 0.0, "location": (0.0, 0.0)})
        }
    
    def residual_momentum_x(
        self,
        x: np.ndarray, y: np.ndarray, t: np.ndarray,
        u: np.ndarray, v: np.ndarray, p: np.ndarray,
        u_t: np.ndarray, u_x: np.ndarray, u_y: np.ndarray,
        u_xx: np.ndarray, u_yy: np.ndarray, p_x: np.ndarray
    ) -> np.ndarray:
        """x-momentum: u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0"""
        return u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
    
    def residual_momentum_y(
        self,
        x: np.ndarray, y: np.ndarray, t: np.ndarray,
        u: np.ndarray, v: np.ndarray, p: np.ndarray,
        v_t: np.ndarray, v_x: np.ndarray, v_y: np.ndarray,
        v_xx: np.ndarray, v_yy: np.ndarray, p_y: np.ndarray
    ) -> np.ndarray:
        """y-momentum: v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0"""
        return v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
    
    def residual_continuity(
        self,
        x: np.ndarray, y: np.ndarray, t: np.ndarray,
        u_x: np.ndarray, v_y: np.ndarray
    ) -> np.ndarray:
        """Continuity: u_x + v_y = 0"""
        return u_x + v_y