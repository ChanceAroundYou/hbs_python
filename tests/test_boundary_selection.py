"""自由边界对应选择器的回归测试。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from hbs.boundary_selection import (
    BoundarySelection,
    _newton_step,
    _parameter_basis,
    _SlidingBoundaryProblem,
    solve_sliding_boundary,
)


def analytic_case(name: str):
    """故意扭曲输入参数的解析族：``γ(t) = exact(σ(t))``。"""

    def sigma(t):
        return t + 0.20 * np.sin(2 * t) + 0.08 * (1 - np.cos(3 * t))

    def dsigma(t):
        return 1 + 0.40 * np.cos(2 * t) + 0.24 * np.sin(3 * t)

    a, b, m = {
        "circle": (1.0, 0.0, 1),
        "ellipse": (0.825, 0.175, 1),
        "nonconvex_laurent": (1.0, 0.35, 2),
    }[name]

    def exact(t):
        return a * np.exp(1j * t) + b * np.exp(-1j * m * t)

    def exact_derivative(t):
        return 1j * a * np.exp(1j * t) - 1j * m * b * np.exp(-1j * m * t)

    curve = lambda t: exact(sigma(t))
    derivative = lambda t: exact_derivative(sigma(t)) * dsigma(t)
    return curve, derivative, exact


def test_parameter_basis_derivative_matches_finite_difference():
    theta = np.linspace(0.0, 2.0 * np.pi, 41, endpoint=False)
    basis, derivative = _parameter_basis(theta, modes=3)
    assert basis.shape == (len(theta), 7)
    assert derivative.shape == basis.shape
    step = 1e-6
    finite_difference = (
        _parameter_basis(theta + step, 3)[0] - _parameter_basis(theta - step, 3)[0]
    ) / (2 * step)
    assert np.allclose(derivative, finite_difference, atol=1e-9)


def test_circle_needs_no_perturbation():
    selection = solve_sliding_boundary(
        lambda t: np.exp(1j * t),
        lambda t: 1j * np.exp(1j * t),
        modes=8,
    )
    assert isinstance(selection, BoundarySelection)
    assert np.max(np.abs(selection.coefficients)) < 1e-12
    assert selection.diagnostics["stop_reason"] == "residual_tolerance"
    assert selection.diagnostics["accepted"] is True
    assert selection.diagnostics["dense_min_parameter_speed"] == pytest.approx(1.0)


def test_parameter_preserves_input_shape():
    selection = solve_sliding_boundary(
        lambda t: np.exp(1j * t),
        lambda t: 1j * np.exp(1j * t),
        modes=4,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    assert selection.parameter(theta).shape == theta.shape
    assert selection.parameter(theta.reshape(3, 4)).shape == (3, 4)


def test_jacobian_matches_finite_difference():
    curve, derivative, _ = analytic_case("ellipse")
    problem = _SlidingBoundaryProblem(curve, derivative, modes=16)
    rng = np.random.default_rng(0)
    initial = 0.002 * rng.normal(size=33) / np.r_[
        1.0, np.arange(1, 17) ** 2, np.arange(1, 17) ** 2
    ]
    direction = rng.normal(size=33)
    direction /= np.linalg.norm(direction)
    step = 2e-6
    state = problem.evaluate(initial, jacobian=True)
    finite_difference = (
        problem.evaluate(initial + step * direction)["residual"]
        - problem.evaluate(initial - step * direction)["residual"]
    ) / (2 * step)
    error = np.linalg.norm(finite_difference - state["jacobian"] @ direction) / max(
        np.linalg.norm(finite_difference), 1e-15
    )
    assert error < 1e-8


def test_analytic_laurent_family_is_recovered():
    curve, derivative, exact = analytic_case("nonconvex_laurent")
    rng = np.random.default_rng(0)
    initial = 0.002 * rng.normal(size=33) / np.r_[
        1.0, np.arange(1, 17) ** 2, np.arange(1, 17) ** 2
    ]
    selection = None
    for modes in (16, 32, 64, 128, 256):
        selection = solve_sliding_boundary(
            curve, derivative, modes=modes, initial=initial, max_iterations=35
        )
        initial = selection.coefficients
        if selection.diagnostics["dense_forbidden_residual"] < 2e-9:
            break
    assert selection is not None
    assert selection.diagnostics["accepted"] is True
    assert selection.diagnostics["dense_forbidden_residual"] < 1e-12
    # 一阶相位现在相对输入曲线自身归一化；解与精确参数化只差一个 Möbius
    # 重参数化（同一外域对应的不同 gauge），所以比较**像集**而不是逐点 p。
    theta = np.linspace(0.0, 2.0 * np.pi, 20000, endpoint=False)
    recovered = curve(selection.parameter(theta))
    reference = exact(np.linspace(0.0, 2.0 * np.pi, 200000, endpoint=False))
    tree = cKDTree(np.c_[reference.real, reference.imag])
    distance, _ = tree.query(np.c_[recovered.real, recovered.imag])
    scale = _SlidingBoundaryProblem(curve, derivative, modes=16).scale
    assert distance.max() / scale < 1e-4


def test_selection_is_rotation_equivariant():
    """整体旋转输入不应改变选出的边界参数化（形状内一阶相位归一化）。"""
    curve, derivative, _ = analytic_case("ellipse")
    base = solve_sliding_boundary(curve, derivative)
    angle = 0.7
    rotated = solve_sliding_boundary(
        lambda t: np.exp(1j * angle) * curve(t),
        lambda t: np.exp(1j * angle) * derivative(t),
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    assert np.allclose(base.parameter(theta), rotated.parameter(theta), atol=1e-10)


def test_auto_continuation_reaches_target_from_zero():
    curve, derivative, _ = analytic_case("nonconvex_laurent")
    selection = solve_sliding_boundary(curve, derivative)  # modes=None: auto
    diagnostics = selection.diagnostics
    assert diagnostics["accepted"] is True
    assert diagnostics["dense_forbidden_residual"] < 2e-7
    assert [stage["modes"] for stage in diagnostics["continuation_stages"]] == [
        8,
        16,
        32,
        64,
        128,
    ]


def test_explicit_modes_stops_at_requested_order():
    curve, derivative, _ = analytic_case("circle")
    selection = solve_sliding_boundary(curve, derivative, modes=16)
    assert selection.modes == 16
    assert [stage["modes"] for stage in selection.diagnostics["continuation_stages"]] == [
        8,
        16,
    ]


def test_initial_disables_automatic_continuation():
    curve, derivative, _ = analytic_case("ellipse")
    selection = solve_sliding_boundary(
        curve, derivative, modes=16, initial=np.zeros(33)
    )
    assert selection.modes == 16
    assert "continuation_stages" not in selection.diagnostics


def test_arbitrary_start_phase_is_initialized():
    """参数起点旋转 π 后，初始一阶系数为负实轴；求解器应自行平移常数相位。"""
    curve, derivative, _ = analytic_case("ellipse")
    shifted = lambda t: curve(t + np.pi)
    shifted_derivative = lambda t: derivative(t + np.pi)
    selection = solve_sliding_boundary(shifted, shifted_derivative)
    assert selection.diagnostics["accepted"] is True
    assert selection.diagnostics["dense_forbidden_residual"] < 2e-7


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        _parameter_basis(np.linspace(0.0, 1.0, 5), modes=0)
    with pytest.raises(ValueError, match="degenerate"):
        _SlidingBoundaryProblem(lambda t: np.ones_like(t), lambda t: np.zeros_like(t))
    problem = _SlidingBoundaryProblem(
        lambda t: np.exp(1j * t), lambda t: 1j * np.exp(1j * t), modes=4
    )
    with pytest.raises(ValueError, match="coefficients"):
        problem.evaluate(np.zeros(4))
    with pytest.raises(ValueError, match="odd length"):
        solve_sliding_boundary(
            lambda t: np.exp(1j * t),
            lambda t: 1j * np.exp(1j * t),
            initial=np.zeros(4),
        )


def test_min_modes_delays_early_stop():
    """残差已过目标但未到 min_modes 时，仍要再加密一档。"""
    curve, derivative, _ = analytic_case("circle")
    selection = solve_sliding_boundary(
        curve, derivative, min_modes=32, target_residual=1e-3
    )
    assert selection.modes == 32
    assert selection.diagnostics["continuation_stages"][-1]["modes"] == 32


def test_min_absolute_residual_allows_early_stop():
    """绝对残差已经极小时，允许在 min_modes 之前停止。"""
    curve = lambda t: np.exp(1j * t)
    derivative = lambda t: 1j * np.exp(1j * t)
    selection = solve_sliding_boundary(
        curve,
        derivative,
        min_modes=64,
        min_absolute_residual=1e-6,
        target_residual=1e-3,
    )
    assert selection.modes == 8
    assert selection.diagnostics["continuation_stages"][-1]["modes"] == 8


def test_min_modes_above_max_modes_is_rejected():
    curve, derivative, _ = analytic_case("circle")
    with pytest.raises(ValueError, match="min_modes"):
        solve_sliding_boundary(
            curve, derivative, max_modes=32, min_modes=64
        )


def test_newton_step_falls_back_on_singular_system():
    """方阵直接求解失败时应回退到最小二乘，而不是把异常漏出去。"""
    delta = _newton_step(np.ones((3, 3)), np.array([1.0, 2.0, 3.0]))
    assert delta.shape == (3,)
    assert np.all(np.isfinite(delta))
