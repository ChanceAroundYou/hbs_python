"""球面共形图与三维 Möbius 变换的单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from hbs.volumetric import RevolutionChart, SpheroidChart, ball_mobius, mobius_scale


def _sphere_points(count=200, seed=0):
    rng = np.random.default_rng(seed)
    unit = rng.normal(size=(count, 3))
    return unit / np.linalg.norm(unit, axis=1)[:, None]


def test_ball_mobius_is_its_own_inverse_with_the_opposite_parameter():
    b = np.array([0.2, -0.1, 0.05])
    unit = _sphere_points()
    np.testing.assert_allclose(ball_mobius(ball_mobius(unit, b), -b), unit, atol=1e-12)


def test_ball_mobius_sends_b_to_the_origin():
    b = np.array([0.3, 0.1, -0.2])
    np.testing.assert_allclose(ball_mobius(b, b), np.zeros(3), atol=1e-14)


def test_mobius_scale_is_positive_and_unit_at_the_center():
    b = np.array([0.2, -0.1, 0.05])
    unit = _sphere_points()
    assert np.all(mobius_scale(unit, b) > 0)
    assert mobius_scale(np.zeros(3), b) == pytest.approx(1 - b @ b)


def test_unit_sphere_chart_is_the_identity():
    chart = SpheroidChart(a=1.0, c=1.0)
    unit = _sphere_points()
    np.testing.assert_allclose(chart.boundary(unit), unit, atol=1e-9)
    assert np.linalg.norm(chart.center_parameter) < 1e-12


def test_spheroid_chart_samples_lie_on_the_ellipsoid():
    chart = SpheroidChart(a=1.0, c=1.5)
    mapped = chart.boundary(_sphere_points())
    implicit = mapped[:, 0] ** 2 + mapped[:, 1] ** 2 + mapped[:, 2] ** 2 / 1.5**2
    np.testing.assert_allclose(implicit, 1.0, atol=1e-9)


def test_spheroid_chart_audit_is_conformal_at_the_tangent_level():
    chart = SpheroidChart(a=1.0, c=1.5)
    audit = chart.audit(count=80)
    assert audit["tangent_ratio_max"] == pytest.approx(1.0, abs=1e-7)
    assert audit["area_center_residual"] < 1e-12
    assert audit["mobius_inverse_error"] < 1e-12


def test_spheroid_chart_target_mobius_stays_on_the_same_conformal_class():
    b = (0.0, 0.0, 0.2)
    pushed = SpheroidChart(a=1.0, c=1.5, target_mobius=b)
    unit = _sphere_points(50)
    back = ball_mobius(pushed.boundary(unit), -np.asarray(b))
    implicit = back[:, 0] ** 2 + back[:, 1] ** 2 + back[:, 2] ** 2 / 1.5**2
    np.testing.assert_allclose(implicit, 1.0, atol=1e-8)
    assert pushed.audit(count=60)["tangent_ratio_max"] == pytest.approx(1.0, abs=1e-7)


def test_revolution_chart_with_unit_radius_is_the_identity():
    chart = RevolutionChart(radius=lambda theta: np.ones_like(theta))
    unit = _sphere_points()
    np.testing.assert_allclose(chart.boundary(unit), unit, atol=1e-9)
    audit = chart.audit(count=80)
    assert audit["tangent_ratio_max"] == pytest.approx(1.0, abs=1e-7)


def test_revolution_chart_rejects_nonpositive_radius():
    with pytest.raises(ValueError, match="strictly positive"):
        RevolutionChart(radius=lambda theta: np.cos(theta), grid=257)


def test_cone_initial_keeps_radial_ratio():
    chart = SpheroidChart(a=1.0, c=2.0)
    assert np.linalg.norm(chart.center_parameter) < 1e-10
    rng = np.random.default_rng(3)
    points = rng.normal(size=(30, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    boundary = chart.boundary(points)
    for radius in (0.0, 0.25, 0.5, 1.0):
        mapped = chart.cone_initial(radius * points)
        np.testing.assert_allclose(mapped, radius * boundary, atol=1e-8)
