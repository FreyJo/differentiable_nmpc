from dataclasses import dataclass

import numpy as np

@dataclass
class LinearDiscreteDynamics:
    """Linear discrete-time dynamics x_{t+1} = A x_t + B u_t + b."""
    A: np.ndarray
    B: np.ndarray
    b: np.ndarray

@dataclass
class QuadraticCost:
    """Quadratic cost 0.5 x^T Q x + 0.5 u^T R u + x^T r + u^T q."""
    Q: np.ndarray
    R: np.ndarray
    r: np.ndarray
    q: np.ndarray

@dataclass
class ControlBounds:
    """Control bounds u_lower <= u <= u_upper."""
    u_lower: np.ndarray
    u_upper: np.ndarray

@dataclass
class ControlBoundedLqrProblem:
    """Linear quadratic regulator problem with control bounds."""
    dynamics: LinearDiscreteDynamics
    cost: QuadraticCost
    control_bounds: ControlBounds
    N_horizon: int

    def __post_init__(self):
        self.nx, self.nu = self.dynamics.A.shape[0], self.dynamics.B.shape[1]
