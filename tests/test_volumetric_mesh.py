"""TetraMesh 容器与结构化构造的单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from hbs.volumetric import (
    TetraMesh,
    as_tetramesh,
    ball_tetra_mesh,
    build_tetramesh,
)

REFERENCE = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


def test_reference_tet_volume_and_p1_gradients_are_exact():
    mesh = build_tetramesh(REFERENCE, [[0, 1, 2, 3]], interior_count=0)
    assert mesh.volume[0] == pytest.approx(1.0 / 6.0, rel=0, abs=1e-15)
    # ∇N_1 = e_x；∇N_0 = -(1,1,1)
    np.testing.assert_allclose(mesh.gradients[0, 1], [1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(mesh.gradients[0, 0], [-1.0, -1.0, -1.0], atol=1e-15)


def test_build_tetramesh_flips_negative_orientation():
    flipped = build_tetramesh(REFERENCE, [[0, 2, 1, 3]], interior_count=0)
    assert flipped.volume[0] > 0


def test_degenerate_tetrahedron_is_rejected():
    flat = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    with pytest.raises(ValueError, match="non-positive volume"):
        build_tetramesh(flat, [[0, 1, 2, 3]], interior_count=0)


def test_ball_mesh_is_deterministic_and_boundary_sits_on_the_sphere():
    first = ball_tetra_mesh(boundary_count=24, interior_count=6, seed=0)
    second = ball_tetra_mesh(boundary_count=24, interior_count=6, seed=0)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.tetra, second.tetra)
    np.testing.assert_array_equal(first.gradients, second.gradients)

    assert np.all(first.volume > 0)
    radii = np.linalg.norm(first.points[first.interior_count :], axis=1)
    np.testing.assert_allclose(radii, 1.0, atol=1e-12)
    assert np.all(np.linalg.norm(first.points[: first.interior_count], axis=1) < 0.68 + 1e-12)
    np.testing.assert_array_equal(first.points[0], np.zeros(3))


def test_ball_mesh_seed_changes_the_mesh():
    first = ball_tetra_mesh(boundary_count=24, interior_count=6, seed=0)
    second = ball_tetra_mesh(boundary_count=24, interior_count=6, seed=1)
    assert not np.array_equal(first.points, second.points)


def test_as_tetramesh_accepts_legacy_dict_and_reproduces_operators():
    mesh = ball_tetra_mesh(boundary_count=16, interior_count=4, seed=2)
    legacy = {
        "points": mesh.points,
        "tetra": mesh.tetra,
        "interior_count": mesh.interior_count,
    }
    rebuilt = as_tetramesh(legacy)
    np.testing.assert_array_equal(rebuilt.gradients, mesh.gradients)
    np.testing.assert_array_equal(rebuilt.volume, mesh.volume)


def test_dict_style_access_includes_custom_keys():
    mesh = ball_tetra_mesh(boundary_count=12, interior_count=3, seed=3)
    mesh.extra["stations"] = np.array([0, 1, 2])
    assert "stations" in mesh
    assert mesh["stations"].tolist() == [0, 1, 2]
    assert mesh.get("missing") is None
    assert mesh.as_dict()["stations"].tolist() == [0, 1, 2]
    assert isinstance(mesh, TetraMesh)
    with pytest.raises(KeyError):
        mesh["missing"]


def test_interior_count_must_be_in_range():
    with pytest.raises(ValueError, match="interior_count"):
        build_tetramesh(REFERENCE, [[0, 1, 2, 3]], interior_count=5)


def test_tetra_indices_must_be_in_range():
    with pytest.raises(ValueError, match="out of range"):
        build_tetramesh(REFERENCE, [[0, 1, 2, 9]], interior_count=0)
