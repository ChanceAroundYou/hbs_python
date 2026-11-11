"""Disk meshes, P1 face fields and initial maps for the fixed-trace solver.

Everything here is ordinary floating point. The mesh is generated fresh from
``spacing`` and ``boundary_count``; no archived mesh is needed.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import Delaunay

TARGET_ANGLE_GAP_FLOOR = 3e-8
TARGET_ANGLE_GAP_CEILING = math.pi/2 - 1e-3


def face_gradients(z, faces):
    """Barycentric gradients and signed areas of a positively oriented mesh."""
    xy = np.c_[z.real, z.imag]
    tri = xy[faces]
    det = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
           - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0]))
    if np.any(det <= 0):
        raise ValueError("source disk mesh must be positively oriented")
    gx = np.stack((tri[:, 1, 1] - tri[:, 2, 1],
                   tri[:, 2, 1] - tri[:, 0, 1],
                   tri[:, 0, 1] - tri[:, 1, 1]), axis=1) / det[:, None]
    gy = np.stack((tri[:, 2, 0] - tri[:, 1, 0],
                   tri[:, 0, 0] - tri[:, 2, 0],
                   tri[:, 1, 0] - tri[:, 0, 0]), axis=1) / det[:, None]
    return gx, gy, 0.5*det


def face_fields(values, faces, gx, gy):
    """P1 field and Wirtinger derivatives from cached face gradients.

    The three vertex gathers avoid materializing the ``(faces, 3)`` complex
    temporary that ``values[faces]`` creates. The arithmetic is the same
    three-term sum, so the result is bit-identical to the old expression.
    """
    fx = (values[faces[:, 0]]*gx[:, 0] + values[faces[:, 1]]*gx[:, 1]
          + values[faces[:, 2]]*gx[:, 2])
    fy = (values[faces[:, 0]]*gy[:, 0] + values[faces[:, 1]]*gy[:, 1]
          + values[faces[:, 2]]*gy[:, 2])
    return 0.5*(fx - 1j*fy), 0.5*(fx + 1j*fy), np.imag(np.conj(fx)*fy)


def face_state(values, faces, gx, gy):
    """Per-face ``|mu|`` and Jacobian."""
    fz, fbar, jacobian = face_fields(values, faces, gx, gy)
    return np.abs(fbar/fz), jacobian


def disk_mesh(spacing=0.1, boundary_count=128):
    """Unstructured disk mesh with every interior vertex free."""
    if spacing <= 0 or boundary_count < 16:
        raise ValueError("spacing must be positive and boundary_count >= 16")
    axis = np.arange(-1.0 + spacing, 1.0, spacing)
    xx, yy = np.meshgrid(axis, axis)
    interior = (xx + 1j*yy).ravel()
    interior = interior[np.abs(interior) < 1.0 - 0.35*spacing]
    theta = 2*np.pi*np.arange(boundary_count)/boundary_count
    z = np.r_[interior, np.exp(1j*theta)]
    faces = Delaunay(np.c_[z.real, z.imag]).simplices.copy()
    xy = np.c_[z.real, z.imag]
    edge1 = xy[faces[:, 1]] - xy[faces[:, 0]]
    edge2 = xy[faces[:, 2]] - xy[faces[:, 0]]
    det = edge1[:, 0]*edge2[:, 1] - edge1[:, 1]*edge2[:, 0]
    flip = det < 0
    faces[flip, 1], faces[flip, 2] = faces[flip, 2].copy(), faces[flip, 1].copy()
    return z, len(interior), theta, faces


def regularized_target_boundary(trace, theta):
    """Choose a robust convex P1 target polygon for the frozen circle trace.

    Some traces put adjacent uniform-source samples less than 1e-8 radians
    apart; their binary64 points can fail convexity checks even though the
    continuous trace is strictly increasing. Only the P1 node angles are
    blended toward the source grid, and the size of that correction is
    reported so a caller can account for it.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or len(theta) < 5 or not np.isfinite(theta).all():
        raise ValueError("finite source boundary angles required")
    raw = np.asarray(trace.angle(theta), dtype=float)
    start = float(raw[0])
    source_step = 2*math.pi/len(theta)
    raw_extended = np.r_[raw, start + 2*math.pi]
    raw_gaps = np.diff(raw_extended)
    if np.any(raw_gaps <= 0) or abs(float(theta[0])) > 1e-12:
        raise ValueError("trace is not an increasing degree-one lift")
    if np.max(np.abs(np.diff(theta) - source_step)) > 1e-10:
        raise ValueError("source mesh boundary must be uniformly sampled")

    mix = 0.0
    for gap in raw_gaps:
        if gap < TARGET_ANGLE_GAP_FLOOR:
            mix = max(mix, (TARGET_ANGLE_GAP_FLOOR - gap)/(source_step - gap))
        if gap > TARGET_ANGLE_GAP_CEILING:
            mix = max(mix, (gap - TARGET_ANGLE_GAP_CEILING)/(gap - source_step))
    if mix > 0:
        mix = min(1.0, mix + max(1e-14, abs(mix)*1e-12))
    source_lift = np.r_[theta + (start - theta[0]),
                        theta[0] + 2*math.pi + (start - theta[0])]
    mixed_extended = (1 - mix)*raw_extended + mix*source_lift
    mixed_gaps = np.diff(mixed_extended)
    if (np.min(mixed_gaps) < TARGET_ANGLE_GAP_FLOOR*(1 - 1e-8)
            or np.max(mixed_gaps) > TARGET_ANGLE_GAP_CEILING*(1 + 1e-8)):
        raise ArithmeticError("regularized target polygon misses its angle-gap limits")
    phases = mixed_extended[:-1]
    residual = phases - raw
    return np.exp(1j*phases), dict(
        method="minimal convexity blend toward the source angular grid",
        mix_fraction=mix,
        max_vertex_phase_correction_radians=float(np.max(np.abs(residual))),
        raw_target_gap_min_radians=float(np.min(raw_gaps)),
        raw_target_gap_max_radians=float(np.max(raw_gaps)),
        regularized_target_gap_min_radians=float(np.min(mixed_gaps)),
        regularized_target_gap_max_radians=float(np.max(mixed_gaps)),
        target_gap_floor_radians=TARGET_ANGLE_GAP_FLOOR,
        target_gap_ceiling_radians=TARGET_ANGLE_GAP_CEILING,
        complete_boundary_after_transfer="the unchanged frozen trace")


def tutte_initial_map(faces, n_vertices, n_interior, boundary_values):
    """Positive-weight graph-harmonic disk embedding.

    Uniform positive edge weights avoid the negative cotangent weights that
    can make a zero-BC harmonic extension fold on concave traces. For a
    triangulated disk with a 3-connected interior graph and a convex boundary
    polygon Tutte's theorem gives a positive embedding; the caller still
    checks every P1 face Jacobian because arbitrary meshes need not meet those
    graph hypotheses.
    """
    triangles = np.asarray(faces, dtype=np.int64)
    edge_pairs = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                 triangles[:, [2, 0]]), axis=0)
    rows = np.r_[edge_pairs[:, 0], edge_pairs[:, 1]]
    cols = np.r_[edge_pairs[:, 1], edge_pairs[:, 0]]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)),
                       shape=(n_vertices, n_vertices)).tocsr()
    graph.data[:] = 1.0
    degree = np.asarray(graph.sum(axis=1)).ravel()
    laplacian = diags(degree, format="csr") - graph
    interior = np.arange(n_interior, dtype=np.int64)
    boundary = np.arange(n_interior, n_vertices, dtype=np.int64)
    if n_interior and np.any(degree[interior] <= 0):
        raise ValueError("Tutte initialization has an isolated interior vertex")
    values = np.empty(n_vertices, dtype=np.complex128)
    values[boundary] = boundary_values
    if n_interior:
        matrix = laplacian[interior][:, interior].tocsc()
        rhs = -(laplacian[interior][:, boundary] @ boundary_values)
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            values[interior] = spsolve(matrix, rhs)
    if not np.isfinite(values).all():
        raise FloatingPointError("Tutte initialization returned non-finite coordinates")
    return values
