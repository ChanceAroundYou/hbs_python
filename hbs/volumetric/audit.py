"""独立核验：离散梯度与多面体边界。

这些函数只做**诊断**，不参与求解路径。它们把解析 Jacobian 与有限差分、把
P1 多面体边界与解析共形图分开核对，避免把「离散梯度写对了」误当成
「连续解对了」。
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial import ConvexHull

from .mesh import TetraMesh, as_tetramesh
from .metrics import deformation_data


def audit_discrete_gradient(
    chart, mesh: TetraMesh | Mapping, seed: int = 515
) -> dict:
    """对三个单元约束（``log K_O``、``log K_I``、``J``）做中心差分核对。

    方向只扰动内部顶点（边界固定）。返回**按 max(1, |解析值|) 缩放后**的最大
    绝对误差；缩放是为了让三个量级不同的约束可以并排比较。
    """
    mesh = as_tetramesh(mesh)
    rng = np.random.default_rng(seed)
    values = np.asarray(chart.cone_initial(mesh.points), dtype=float)
    direction = rng.normal(size=values.shape)
    direction[mesh.interior_count :] = 0.0
    direction /= np.linalg.norm(direction)

    _, U, s, Vh, J, _, _ = deformation_data(values, mesh)
    invT = np.einsum("tik,tk,tkj->tij", U, 1 / s, Vh)
    dO = (
        3 * np.einsum("ti,tj->tij", U[:, :, 0], Vh[:, 0, :]) / s[:, 0, None, None]
        - invT
    )
    dI = (
        invT
        - 3 * np.einsum("ti,tj->tij", U[:, :, -1], Vh[:, -1, :]) / s[:, -1, None, None]
    )
    dJ = J[:, None, None] * invT
    dA = np.einsum("tvi,tvj->tij", direction[mesh.tetra], mesh.gradients)
    analytic = np.stack(
        [np.einsum("tij,tij->t", d, dA) for d in (dO, dI, dJ)]
    )

    h = 2e-7
    samples = []
    for sign in (1, -1):
        *_, jj, oo, ii = deformation_data(values + sign * h * direction, mesh)
        samples.append(np.stack([np.log(oo), np.log(ii), jj]))
    difference = (samples[0] - samples[1]) / (2 * h)
    absolute = np.max(np.abs(difference - analytic), axis=1)
    scaled = absolute / np.maximum(1.0, np.max(np.abs(analytic), axis=1))
    return dict(
        log_KO_scaled_error=float(scaled[0]),
        log_KI_scaled_error=float(scaled[1]),
        J_scaled_error=float(scaled[2]),
        step=h,
    )


def audit_polyhedral_boundary(chart, mesh: TetraMesh | Mapping, volume_values=None) -> dict:
    """把解析共形图与 P1 多面体边界分开核对。

    检查项包括：边界三角形上的切向奇异值比、边中点误差（相对包围盒对角线）、
    根盖的径向行列式符号、立体角总和、以及多面体边界体积与四面体体积是否一致。
    """
    mesh = as_tetramesh(mesh)
    source = mesh.points
    target = np.asarray(chart.boundary(source[mesh.interior_count :]), dtype=float)

    values = np.zeros_like(source)
    values[mesh.interior_count :] = target
    faces = ConvexHull(source).simplices
    orientation = np.einsum(
        "ti,ti->t",
        source[faces[:, 0]],
        np.cross(source[faces[:, 1]], source[faces[:, 2]]),
    )
    flip = orientation < 0
    faces[flip, 1], faces[flip, 2] = faces[flip, 2].copy(), faces[flip, 1].copy()

    ds = np.transpose(source[faces[:, 1:]] - source[faces[:, 0, None]], (0, 2, 1))
    dt = np.transpose(values[faces[:, 1:]] - values[faces[:, 0, None]], (0, 2, 1))
    basis = np.stack(
        [
            ds[:, :, 0] / np.linalg.norm(ds[:, :, 0], axis=1)[:, None],
            np.zeros((len(ds), 3)),
        ],
        axis=-1,
    )
    temp = ds[:, :, 1] - np.sum(ds[:, :, 1] * basis[:, :, 0], axis=1)[:, None] * basis[:, :, 0]
    basis[:, :, 1] = temp / np.linalg.norm(temp, axis=1)[:, None]
    local = np.einsum("tki,tkj->tij", basis, ds)
    derivative = np.einsum("tik,tkj->tij", dt, np.linalg.inv(local))
    s = np.linalg.svd(derivative, compute_uv=False)

    edges = np.unique(
        np.sort(
            np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1
        ),
        axis=0,
    )
    midpoint = (source[edges[:, 0]] + source[edges[:, 1]]) / 2
    midpoint /= np.linalg.norm(midpoint, axis=1)[:, None]
    exact = chart.boundary(midpoint)
    linear = (values[edges[:, 0]] + values[edges[:, 1]]) / 2
    extent = float(np.linalg.norm(np.ptp(target, axis=0)))
    center = chart.cone_initial(np.zeros((1, 3)))[0]

    face_vectors = values[faces] - center
    aa, bb, cc = np.moveaxis(face_vectors, 1, 0)
    triple = np.einsum("ti,ti->t", aa, np.cross(bb, cc))
    lengths = np.linalg.norm(face_vectors, axis=2)
    denominator = (
        np.prod(lengths, axis=1)
        + np.einsum("ti,ti->t", aa, bb) * lengths[:, 2]
        + np.einsum("ti,ti->t", bb, cc) * lengths[:, 0]
        + np.einsum("ti,ti->t", cc, aa) * lengths[:, 1]
    )
    solid_angle = 2 * np.arctan2(triple, denominator)
    boundary_volume = float(
        np.sum(
            np.einsum(
                "ti,ti->t",
                values[faces[:, 0]],
                np.cross(values[faces[:, 1]], values[faces[:, 2]]),
            )
        )
        / 6
    )
    if volume_values is None:
        volume_values = chart.cone_initial(source)
    _, _, _, _, J, _, _ = deformation_data(volume_values, mesh)
    tetra_volume = float(np.sum(mesh.volume * J))
    return dict(
        P1_boundary_tangent_ratio_max=float(np.max(s[:, 0] / s[:, 1])),
        edge_midpoint_error_over_bbox_diagonal=float(
            np.max(np.linalg.norm(exact - linear, axis=1)) / extent
        ),
        source_min_tetra_volume=float(np.min(mesh.volume)),
        target_star_center=center.tolist(),
        all_boundary_radial_determinants_positive=bool(np.all(triple > 0)),
        minimum_boundary_radial_determinant=float(np.min(triple)),
        minimum_normalized_boundary_radial_determinant=float(
            np.min(triple / np.prod(lengths, axis=1))
        ),
        solid_angle_degree=float(np.sum(solid_angle) / (4 * np.pi)),
        oriented_polyhedral_boundary_volume=boundary_volume,
        mapped_tetrahedral_volume=tetra_volume,
        relative_boundary_tetra_volume_mismatch=float(
            abs(boundary_volume - tetra_volume) / abs(boundary_volume)
        ),
    )
