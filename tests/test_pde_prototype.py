"""实验性 PDE 路径：内域平坦度量 + 外域 Fourier 选择。"""

from __future__ import annotations

import numpy as np
import pytest

from hbs._engines import ENGINES
from hbs.harmonic_measure import harmonic_measure_welding
from hbs.pde import pde_welding, solve_metric_trace


def _ellipse(n=500, a=1.2, b=0.8):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], 1)


def test_pde_is_registered():
    assert "pde" in ENGINES


def test_pde_engine_agrees_with_harmonic_measure():
    """M1 居中后，pde 与 harmonic_measure 只差源圆旋转。"""
    from hbs import cache, get_hbs_result
    from hbs._pipeline import compare_engines

    t = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    z = (1 + 0.30 * np.cos(t - 1.0)) * np.exp(1j * t)
    bound = np.stack([z.real, z.imag], 1)
    disk, fem = cache.load_unit_disk(0.05, 120)
    pde = get_hbs_result(
        bound, circle_point_num=120, density=0.05, disk=disk,
        method="pde", normalize="none",
    )
    reference = get_hbs_result(
        bound, circle_point_num=120, density=0.05, disk=disk,
        method="harmonic_measure", normalize="none",
    )
    agreement = compare_engines(pde.welding, reference.welding, disk, fem, 120)
    assert np.degrees(agreement.welding_rms) < 1.0
    assert agreement.mu_rms < 0.1


def test_pde_direct_solver_rejects_nonconvex_hard_oval():
    """关闭几何延续时，强非凸边界仍必须显式失败，不静默降级。"""
    from hbs._pipeline import prepare_boundary
    from hbs.pde import pde_welding

    t = np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    z = np.cos(t) + 0.3 * np.cos(2 * t) + 1j * (np.sin(t) + 0.2 * np.sin(3 * t))
    bound = np.stack([z.real, z.imag], 1)
    prepared = prepare_boundary(bound, 200, resample=False)
    with pytest.raises(RuntimeError, match="PDE interior"):
        pde_welding(prepared, circle_num=200, continuation=False)


def test_pde_welding_is_rotation_covariant():
    """整体旋转输入只改变焊接的整体相位，不改变相对结构。"""
    bound = _ellipse()
    angle = 0.7
    rotated = np.stack(
        [
            np.cos(angle) * bound[:, 0] - np.sin(angle) * bound[:, 1],
            np.sin(angle) * bound[:, 0] + np.cos(angle) * bound[:, 1],
        ],
        1,
    )
    base = np.asarray(pde_welding(bound, circle_num=120, n=96)["init_y"], complex)
    moved = np.asarray(pde_welding(rotated, circle_num=120, n=96)["init_y"], complex)
    deviation = np.angle(moved / base)
    deviation -= np.angle(np.mean(np.exp(1j * deviation)))
    assert np.degrees(np.sqrt(np.mean(deviation**2))) < 1e-9


def test_metric_trace_meets_raw_acceptance():
    from hbs.pde import _BoundaryShape

    shape = _BoundaryShape(_ellipse())
    u, alpha, diagnostics = solve_metric_trace(shape, n=96)
    assert diagnostics["geometry_pde_accepted"] is True
    assert diagnostics["curvature_robin_max"] < 1e-6
    assert diagnostics["normalization_residual"] < 1e-8
    assert diagnostics["augmentation_norm"] < 1e-6
    assert u.shape == alpha.shape == (96,)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(alpha))
    assert abs(alpha[0]) < 1e-12


def test_pde_welding_is_monotone_and_close_to_harmonic_measure():
    bound = _ellipse()
    welding = pde_welding(bound, circle_num=120, n=96)
    y = np.asarray(welding["init_y"], complex)
    assert y.shape == (120,)
    assert np.all(np.isfinite(y))
    assert np.allclose(np.abs(y), 1.0)
    assert np.all(np.diff(np.unwrap(np.angle(y))) > 0)

    reference = harmonic_measure_welding(bound, circle_num=120, n=1024)["init_y"]
    deviation = np.angle(reference / y)
    deviation -= np.angle(np.mean(np.exp(1j * deviation)))
    rms_degrees = np.degrees(np.sqrt(np.mean(deviation**2)))
    assert rms_degrees < 1.0, f"PDE 与调和测度焊接差 {rms_degrees:.3f}°"


def test_pde_welding_rejects_non_smooth_input():
    # 精确三尖点：|γ'| 在尖点为零，内域 PDE 不应假装成功。
    t = np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    bound = np.stack(
        [np.cos(t) + 0.5 * np.cos(2 * t + 0.6), np.sin(t) - 0.5 * np.sin(2 * t + 0.6)],
        1,
    )
    with pytest.raises((ValueError, RuntimeError)):
        pde_welding(bound, circle_num=120, n=96)


def test_pde_warm_start_reduces_iterations():
    """n>=256 时先用 n=96 解再插值，内域迭代数应下降。"""
    bound = _ellipse()
    cold = pde_welding(bound, circle_num=120, n=256, warm_start=False)
    warm = pde_welding(bound, circle_num=120, n=256, warm_start=True)
    assert cold["pde_diagnostics"]["warm_start_n"] is None
    assert warm["pde_diagnostics"]["warm_start_n"] == 96
    assert warm["pde_diagnostics"]["iterations"] < cold["pde_diagnostics"]["iterations"]
    assert warm["pde_diagnostics"]["curvature_robin_max"] < 1e-6


def test_pde_selector_uses_min_mode_policy():
    """PDE 侧选择器不因低阶残差先过 3e-4 就在第 8 阶停。"""
    t = np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    z = (1 + 0.30 * np.cos(3 * t + 0.5) + 0.12 * np.cos(t - 0.9)) * np.exp(1j * t)
    bound = np.stack([z.real, z.imag], 1)
    trace = pde_welding(bound, circle_num=120, n=96)
    selection = trace["selection_diagnostics"]
    assert selection["modes"] <= 128
    assert selection["dense_forbidden_residual"] <= 3e-4
    assert trace["pde_diagnostics"]["selection_gate"] == 3e-4
