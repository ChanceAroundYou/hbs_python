"""Callable fixed-complete-circle TBS solver.

    >>> import numpy as np
    >>> from hbs.tbs import solve_tbs
    >>> beta = np.linspace(0, 2*np.pi, 257)[:-1]
    >>> alpha = beta + 0.1*np.sin(2*beta)
    >>> result = solve_tbs(beta, alpha, boundary_count=128)
    >>> result["final_k"] < result["start_k"]
    True

Import surface kept deliberately small: boundary inputs, mesh construction,
the P1 minimax solver. Certificates (upper interval enclosure, lower RS
bound, acceptance replay) are not part of this package.
"""
from .boundary import FourierCircleTrace, LocalC2Boundary
from .geometry import (disk_mesh, face_fields, face_gradients, face_state,
                       regularized_target_boundary, tutte_initial_map)
from .solver import LSE_BETAS, solve_tbs

__all__ = [
    "solve_tbs", "LSE_BETAS", "LocalC2Boundary", "FourierCircleTrace", "disk_mesh",
    "regularized_target_boundary", "tutte_initial_map",
    "face_gradients", "face_fields", "face_state",
]
