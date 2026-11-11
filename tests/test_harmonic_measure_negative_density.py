"""近角边界的配点阶梯与负密度接受判据。

背景：真实轮廓上的原始调和测度密度会出现 −1e-6～−1e-9 的负值。诊断表明它
随配点数下降（负质量分数约 n^-2），属于离散化伪影。这里固定四条契约：

1. 光滑椭圆在 n=1024 仍然严格正、零修正质量（默认路径不变）；
2. 近角星形在 n=1024 失败，经配点阶梯在更高 n 接受，负质量 ≤1e-8；
3. 接受同时受**峰比**约束（最深单点负值 / 正峰 ≤1e-6），只查质量会把几何退化
   当伪影救回来；
4. 尖点输入在整个阶梯上仍被拒绝（负值不随 n 消失，看似干净的那一级其 2n 参照
   达不到判据，记为未收敛）。

阶梯的作用范围在 `tests/test_engines_ladder_scope.py`。
"""
from __future__ import annotations

import numpy as np
import pytest

import hbs._harmonic_measure.direct as direct_mod
from hbs._harmonic_measure.direct import DirectHBS
from hbs.harmonic_measure import harmonic_measure_welding


def _ellipse(n: int = 800) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    z = 1.3 * np.cos(t) + 1j * 0.8 * np.sin(t)
    return np.c_[z.real, z.imag]


def _star(k: int = 7, a: float = 0.7, n: int = 2000) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    z = (1 + a * np.cos(k * t)) * np.exp(1j * t)
    return np.c_[z.real, z.imag]


def test_ellipse_stays_strict_at_first_level():
    out = harmonic_measure_welding(_ellipse(), circle_num=200, n_ladder=(1024,))
    assert out["n"] == 1024
    assert out["density_acceptance"] == "strict"
    assert out["negative_mass_internal"] == 0.0
    assert out["negative_mass_external"] == 0.0


def test_star_needs_ladder_and_accepts_higher_n():
    out = harmonic_measure_welding(
        _star(), circle_num=200, n_ladder=(1024, 2048, 4096),
        check_convergence=False,
    )
    assert out["n"] >= 2048
    assert out["negative_mass_internal"] <= 1e-8
    assert out["negative_mass_external"] <= 1e-8
    y = np.unwrap(np.angle(out["init_y"]))
    assert np.all(np.diff(y) > 0)


def test_strict_tolerance_rejects_star_at_all_levels():
    with pytest.raises(RuntimeError, match="negative"):
        harmonic_measure_welding(
            _star(), circle_num=200, n_ladder=(1024, 2048),
            negative_mass_tolerance=0.0, check_convergence=False,
        )


def _cusped(amplitude: float = 0.5, count: int = 500) -> np.ndarray:
    """`z(t) = e^{it} + a e^{-i(2t+0.6)}`; `a = 0.5` is an exact three-cusp curve."""
    t = 2 * np.pi * np.arange(count) / count
    return np.stack(
        [np.cos(t) + amplitude * np.cos(2 * t + 0.6),
         np.sin(t) - amplitude * np.sin(2 * t + 0.6)],
        1,
    )


def test_peak_ratio_is_reported_for_an_accepted_trace():
    out = harmonic_measure_welding(
        _star(), circle_num=200, n_ladder=(1024, 2048, 4096),
        check_convergence=False,
    )
    assert out["n"] >= 2048
    assert out["density_acceptance"] in {"strict", "negative_mass"}
    assert out["negative_peak_ratio_internal"] <= 1e-6
    assert out["negative_peak_ratio_external"] <= 1e-6


def test_cusped_boundary_is_rejected_across_the_ladder():
    """An exact cusp is not rescued by refining the collocation grid.

    Measured (500-point input, n = 1024 / 2048 / 4096 / 8192): negative mass
    5.2e-3 / 1.6e-2 / ~0 / 5.3e-3 with peak ratios 0.19 / 0.58 / ~0 / 0.34.  The
    negativity does **not** decay with ``n``, and the one level that looks clean is
    only clean by accident: its 8192-point reference does not reach the criteria,
    so that level is recorded as unconverged rather than accepted.
    """
    with pytest.raises(RuntimeError) as exc:
        harmonic_measure_welding(
            _cusped(), circle_num=500, n_ladder=(1024, 2048, 4096, 8192),
        )
    message = str(exc.value)
    assert "peak ratio" in message
    assert "rejected_unconverged" in message


def test_acceptance_contract_requires_both_bounds():
    """The gate is a conjunction: mass alone accepts what the peak bound rejects."""
    from hbs._harmonic_measure.direct import _density_tolerances_are_exceeded

    roundoff = (-1e-13, 1.0, 1e-13, 1e-13)
    artifact = (-1e-7, 1.0, 1e-10, 1e-7)          # mass 1e-10, peak ratio 1e-7
    degenerate = (-1.0, 1.0, 5e-9, 1.0)           # tiny mass, O(1) deepest point

    assert not _density_tolerances_are_exceeded(*roundoff, 1e-8, 1e-6)
    assert not _density_tolerances_are_exceeded(*artifact, 1e-8, 1e-6)
    assert _density_tolerances_are_exceeded(*degenerate, 1e-8, 1e-6)
    # with the peak bound disabled the degenerate case would have been accepted
    assert not _density_tolerances_are_exceeded(*degenerate, 1e-8, float("inf"))


def test_ladder_accepts_small_negative_mass(monkeypatch):
    seen = []

    def fake_compute(curve, derivative, n=480, pole=0j, negative_mass_tolerance=0.0,
                     negative_peak_tolerance=0.0):
        seen.append((n, negative_mass_tolerance))
        mass = {1024: 1e-6, 2048: 1e-9}[n]
        if negative_mass_tolerance < mass:
            raise RuntimeError(f"negative density {mass:.1e} beyond tolerance")
        t = 2 * np.pi * np.arange(n) / n
        return DirectHBS(
            boundary=np.exp(1j * t), alpha=t.copy(), beta=t.copy(),
            rho_internal=np.ones(n), rho_external=np.ones(n),
            mobius_center=0j, internal_equation_residual=0.0,
            external_equation_residual=0.0, timings={},
            min_raw_internal_density=-1e-9, min_raw_external_density=-1e-9,
            density_regularization_mass=mass,
            negative_mass_internal=mass, negative_mass_external=0.0,
        )

    monkeypatch.setattr(direct_mod, "compute_direct_hbs", fake_compute)
    out = harmonic_measure_welding(
        _ellipse(), circle_num=100, n_ladder=(1024, 2048),
        check_convergence=False,
    )
    assert seen == [(1024, 1e-8), (2048, 1e-8)]
    assert out["n"] == 2048
    assert out["density_acceptance"] == "negative_mass"
    assert out["negative_mass_internal"] == pytest.approx(1e-9)
