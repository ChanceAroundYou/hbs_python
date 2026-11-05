"""Tests for the public Riemann-map tool."""

from __future__ import annotations

import numpy as np
import pytest

from hbs import RiemannMap, riemann_map


def _f0(z):
    return z + 0.15 * z**3


def _df0(z):
    return 1.0 + 0.45 * z**2


def _psi(s):
    return s + 0.30 * np.sin(3.0 * s)


def _dpsi(s):
    return 1.0 + 0.90 * np.cos(3.0 * s)


def _curve(s):
    s = np.asarray(s, float)
    return _f0(np.exp(1j * _psi(s)))


def _derivative(s):
    s = np.asarray(s, float)
    e = np.exp(1j * _psi(s))
    return _df0(e) * 1j * e * _dpsi(s)


def _sample_points(radius: float = 0.75) -> np.ndarray:
    r = np.linspace(0.0, radius, 7)
    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    R, T = np.meshgrid(r, theta, indexing="ij")
    return (R * np.exp(1j * T)).ravel()


def test_disk_map_is_identity():
    theta = 2.0 * np.pi * np.arange(256) / 256
    bound = np.column_stack([np.cos(theta), np.sin(theta)])
    result = RiemannMap.from_boundary(bound, n=256)

    points = _sample_points(0.8)
    assert np.max(np.abs(result(points) - points)) < 1e-10
    assert np.max(np.abs(result.derivative(points) - 1.0)) < 1e-9
    assert abs(result.diagnostics.f_at_zero) < 1e-10
    assert result.diagnostics.f_prime_at_zero.real > 0.0
    assert abs(result.diagnostics.f_prime_at_zero.imag) < 1e-10


def test_known_polynomial_map_with_reparameterized_boundary():
    result = RiemannMap.from_curve(_curve, _derivative, n=512)
    points = _sample_points(0.75)
    exact = _f0(points)

    assert np.max(np.abs(result(points) - exact)) < 1e-6
    assert np.max(np.abs(result.derivative(points) - _df0(points))) < 1e-6
    assert abs(result.diagnostics.f_at_zero) < 1e-7
    assert result.diagnostics.min_abs_derivative > 0.5
    residual = result.cauchy_riemann_residual(grid=41, radius=0.6)
    assert residual["max_abs_Fzbar"] < 1e-9
    assert residual["max_abs_mu"] < 1e-9


def test_clockwise_curve_is_reoriented():
    def curve_cw(s):
        return _curve(2.0 * np.pi - np.asarray(s, float))

    def derivative_cw(s):
        return -_derivative(2.0 * np.pi - np.asarray(s, float))

    result = RiemannMap.from_curve(curve_cw, derivative_cw, n=512)
    points = _sample_points(0.7)
    assert np.max(np.abs(result(points) - _f0(points))) < 1e-6


def test_boundary_samples_with_nonuniform_parameterization():
    u = np.linspace(0.0, 1.0, 512, endpoint=False)
    s = 2.0 * np.pi * (u + 0.08 * np.sin(2.0 * np.pi * u))
    boundary = _curve(s)
    bound = np.column_stack([boundary.real, boundary.imag])
    result = RiemannMap.from_boundary(bound, n=512)
    points = _sample_points(0.7)
    assert np.max(np.abs(result(points) - _f0(points))) < 1e-4


def test_source_rotation_makes_derivative_positive_real():
    def curve(s):
        return 1.5 + _curve(s)

    def derivative(s):
        return _derivative(s)

    result = RiemannMap.from_curve(curve, derivative, n=512)
    derivative_at_zero = result.derivative(0j)
    assert abs(derivative_at_zero.imag) < 1e-8 * abs(derivative_at_zero)
    assert derivative_at_zero.real > 0.0
    assert abs(result.diagnostics.f_at_zero - 1.5) < 1e-7


def test_riemann_map_convenience_wrapper():
    theta = 2.0 * np.pi * np.arange(128) / 128
    bound = np.column_stack([1.3 * np.cos(theta), 0.8 * np.sin(theta)])
    result = riemann_map(bound, n=128)
    assert isinstance(result, RiemannMap)
    assert np.all(np.isfinite(result.boundary))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        RiemannMap.from_boundary(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        RiemannMap.from_curve(lambda s: np.zeros(2), lambda s: np.ones(2))
    with pytest.raises(ValueError):
        RiemannMap.from_curve(_curve, _derivative, n=255)
    with pytest.raises(TypeError):
        RiemannMap.from_curve("not callable", _derivative)
