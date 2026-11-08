"""广义 HBS 的边界 welding 与物理域 Dirichlet 延拓测试。"""

from __future__ import annotations

import numpy as np
import pytest

from hbs.generalized_hbs import (
    DomainChart,
    boundary_welding,
    solve_generalized_hbs,
)


def _circle(n: int = 256, radius: float = 1.0) -> np.ndarray:
    t = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def _ellipse(n: int = 256, a: float = 1.25, b: float = 0.8) -> np.ndarray:
    t = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([a * np.cos(t), b * np.sin(t)])


def _asymmetric(n: int = 512) -> np.ndarray:
    t = 2.0 * np.pi * np.arange(n) / n
    radius = 1.0 + 0.2 * np.cos(3.0 * t) + 0.07 * np.sin(2.0 * t + 0.4)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def _asymmetric2(n: int = 512) -> np.ndarray:
    t = 2.0 * np.pi * np.arange(n) / n
    radius = 1.08 + 0.26 * np.cos(3.0 * t + 0.2) + 0.05 * np.sin(2.0 * t)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def _similarity_z(z, scale: float = 1.7, angle: float = 0.4,
                  shift: complex = 0.3 - 0.2j) -> np.ndarray:
    return scale * np.exp(1j * angle) * np.asarray(z, complex) + shift


def _move_boundary(points: np.ndarray, angle: float = 0.4) -> np.ndarray:
    z = points[:, 0] + 1j * points[:, 1]
    moved = _similarity_z(z, angle=angle)
    return np.column_stack([moved.real, moved.imag])


def _chart_map(chart: DomainChart, theta: np.ndarray) -> np.ndarray:
    beta = np.r_[chart.beta, chart.beta[0] + 2.0 * np.pi]
    alpha = np.r_[chart.alpha, chart.alpha[0] + 2.0 * np.pi]
    return np.exp(1j * np.interp(np.mod(theta, 2.0 * np.pi), beta, alpha))


def test_identity_boundary_welding_returns_the_base_boundary():
    points = _ellipse()
    welding = boundary_welding(points, points, n=256)

    assert welding.values.shape == (256,)
    assert np.max(np.abs(welding.values - welding.base_points)) < 1e-3


def test_welding_is_orientation_preserving_and_periodically_closed():
    welding = boundary_welding(_ellipse(), _circle(), n=256)
    t = welding.target_parameter
    increments = np.diff(np.r_[t, t[0] + 2.0 * np.pi])

    assert np.all(increments > 0.0)
    assert np.all(np.isfinite(welding.values))


def test_disk_base_welding_matches_the_target_parent_map():
    """B = D 时 F 与目标父映射只差允许保留的残余旋转。"""
    target = _ellipse()
    chart = DomainChart.from_boundary(target, n=256)
    welding = boundary_welding(_circle(), target, n=256)
    expected = chart.result.boundary_map(welding.t)

    phase = np.angle(np.sum(np.conj(expected) * welding.values))
    residual = np.abs(welding.values - expected * np.exp(1j * phase))

    assert np.max(residual) < 1e-6


def test_similarity_covariance_of_the_welding():
    """非对称形状上，旋转+缩放+平移协变逐点成立。"""
    base, target = _asymmetric(), _asymmetric2()
    moved = boundary_welding(_move_boundary(base), _move_boundary(target), n=256)
    reference = _similarity_z(boundary_welding(base, target, n=256).values)

    assert np.max(np.abs(moved.values - reference)) < 1e-6


def test_m1_centering_leaves_zero_boundary_barycenter():
    chart = DomainChart.from_boundary(_ellipse(), n=256)
    base_points = np.exp(1j * chart.alpha)

    assert abs(np.mean(base_points)) < 1e-6


def test_identity_pair_has_near_zero_beltrami():
    pytest.importorskip("triangle")
    points = _ellipse()
    result = solve_generalized_hbs(points, points, n=256, h=0.05)

    mu = np.abs(result.mu_raw)
    assert np.all(np.isfinite(mu))
    assert result.boundary_order_preserved
    assert float(np.average(mu, weights=np.abs(result.mesh.area))) < 0.1
    assert result.nonpositive_jacobian == 0


def test_reversed_boundary_orientation_is_accepted():
    points = _ellipse()[::-1]
    welding = boundary_welding(points, _circle(), n=256)

    assert np.all(np.isfinite(welding.values))
    assert np.all(np.diff(np.r_[welding.target_parameter,
                                 welding.target_parameter[0] + 2.0 * np.pi]) > 0.0)


def test_welding_is_invariant_to_the_target_sample_roll():
    """只滚动目标的输入起点不应改变几何 welding。"""
    base, target = _asymmetric(), _ellipse()
    reference = boundary_welding(base, target, n=256)
    rolled = boundary_welding(base, np.roll(target, 37, axis=0), n=256)

    assert np.max(np.abs(rolled.values - reference.values)) < 1e-6


def test_degenerate_boundary_is_rejected():
    bad = np.zeros((64, 2))
    with pytest.raises(ValueError):
        DomainChart.from_boundary(bad, n=64)


def test_pde_engine_chart_matches_harmonic_measure_modulo_source_rotation():
    """PDE 相位引擎与调和测度在源圆旋转意义下一致。"""
    points = _asymmetric(256)
    hm = DomainChart.from_boundary(points, n=128)
    pde = DomainChart.from_boundary(points, n=128, engine="pde")
    theta = 2.0 * np.pi * np.arange(257) / 257
    candidate = _chart_map(pde, theta)

    best = None
    for shift in np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False):
        shifted = _chart_map(hm, theta + shift)
        phase = np.angle(np.sum(np.conj(shifted) * candidate))
        residual = float(np.max(np.abs(candidate - shifted * np.exp(1j * phase))))
        if best is None or residual < best[0]:
            best = (residual, shift, phase)

    assert best[0] < 2e-2
    assert np.all(np.isfinite(candidate))
    assert np.all(np.diff(np.r_[pde.beta, pde.beta[0] + 2.0 * np.pi]) > 0.0)
    assert np.all(np.diff(np.r_[pde.alpha, pde.alpha[0] + 2.0 * np.pi]) > 0.0)


def test_unknown_welding_engine_is_rejected():
    with pytest.raises(ValueError):
        DomainChart.from_boundary(_ellipse(128), n=64, engine="not-an-engine")


def test_unknown_rung_is_rejected():
    with pytest.raises(ValueError):
        DomainChart.from_boundary(_ellipse(128), n=64, rung="not-a-rung")


def test_auto_rung_keeps_smooth_shapes_on_raw():
    chart = DomainChart.from_boundary(
        _asymmetric(256), n=128, engine="auto", rung="auto")
    assert chart.engine == "harmonic_measure"
    assert chart.rung == "raw"
