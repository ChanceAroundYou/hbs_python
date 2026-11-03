"""三维畸变量 K 的单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from hbs.volumetric import (
    ball_tetra_mesh,
    boundary_vertex_k,
    build_tetramesh,
    deformation_data,
    distortion_metrics,
    element_k,
)

REFERENCE = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


def _single_tet():
    return build_tetramesh(REFERENCE, [[0, 1, 2, 3]], interior_count=0)


def test_identity_has_unit_distortion():
    mesh = _single_tet()
    K = element_k(REFERENCE, mesh)
    np.testing.assert_allclose(K, 1.0, atol=1e-14)
    metrics = distortion_metrics(REFERENCE, mesh)
    assert metrics["max_K"] == pytest.approx(1.0, abs=1e-14)
    assert metrics["min_J"] == pytest.approx(1.0, abs=1e-14)
    assert metrics["negative_or_zero_tets"] == 0
    assert metrics["weighted_mean_K"] == pytest.approx(1.0, abs=1e-14)
    assert metrics["weighted_p95_K"] == pytest.approx(1.0, abs=1e-14)


def test_uniform_scaling_is_not_distortion_but_changes_jacobian():
    mesh = _single_tet()
    values = 2.0 * REFERENCE
    _, _, sigma, _, J, KO, KI = deformation_data(values, mesh)
    np.testing.assert_allclose(sigma, 2.0, atol=1e-14)
    np.testing.assert_allclose(J, 8.0, atol=1e-14)
    np.testing.assert_allclose(element_k(values, mesh), 1.0, atol=1e-14)
    assert np.all(KO > 0) and np.all(KI > 0)


def test_rotation_is_not_distortion():
    mesh = _single_tet()
    angle = np.pi / 2
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    values = REFERENCE @ rotation.T
    np.testing.assert_allclose(element_k(values, mesh), 1.0, atol=1e-14)
    # 但绕 z 轴旋转会把参考四面体变成负向？行列式仍为 +1。
    assert deformation_data(values, mesh)[4][0] > 0


def test_shear_raises_k_while_keeping_positive_jacobian():
    mesh = _single_tet()
    values = REFERENCE.copy()
    values[:, 0] += 0.5 * REFERENCE[:, 1]
    K = element_k(values, mesh)
    assert np.all(K > 1.0)
    assert deformation_data(values, mesh)[4][0] > 0


def test_reflected_tet_has_nonpositive_jacobian():
    mesh = _single_tet()
    values = REFERENCE.copy()
    values[1, 0] = -1.0  # 翻面
    metrics = distortion_metrics(values, mesh)
    assert metrics["min_J"] <= 0
    assert metrics["negative_or_zero_tets"] == 1


def test_values_shape_is_validated():
    mesh = _single_tet()
    with pytest.raises(ValueError, match="shape"):
        deformation_data(REFERENCE[:3], mesh)


def test_boundary_vertex_k_is_unit_for_the_identity_map():
    mesh = ball_tetra_mesh(boundary_count=16, interior_count=4, seed=0)
    values = mesh.points.copy()
    K = boundary_vertex_k(values, mesh)
    assert K.shape == (len(mesh.points) - mesh.interior_count,)
    np.testing.assert_allclose(K, 1.0, atol=1e-12)


def test_boundary_vertex_k_is_a_volume_weighted_average():
    mesh = ball_tetra_mesh(boundary_count=16, interior_count=4, seed=0)
    values = mesh.points.copy() * 1.2  # 均匀缩放：K 仍为 1
    np.testing.assert_allclose(boundary_vertex_k(values, mesh), 1.0, atol=1e-12)

    values = mesh.points.copy()
    values[:, 0] += 0.2 * mesh.points[:, 1]
    face_K = boundary_vertex_k(values, mesh)
    element = element_k(values, mesh)
    assert np.all(face_K >= element.min() - 1e-12)
    assert np.all(face_K <= element.max() + 1e-12)
