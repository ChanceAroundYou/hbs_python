"""`hbs.weld_compare` 的契约测试：两条焊接的一致性度量。

分两层：

1. **数学层**（`analytic_welding`）：用解析相位构造焊接，与采样密度无关。
   构造已知源圆旋转 `f₂(θ) = f₁(θ+T₀) - f₁(T₀)`，度量必须原样恢复 `T₀` 且
   报告的本质差距到机器精度。这层钉住两个易错点：
   - 源圆旋转是 `f(θ+T) - f(T)`，不是 `f(θ-T)`；
   - `δ` 必须先不 wrap 地算出来再做统计，否则 `S` 会污染 `rms`。
2. **实现层**：真实引擎的焊接是**分段线性**的，两条不同采样网格之间比较时
   有 O(h²) 的插值底噪，所以该层的容差按离散误差给，不按机器精度给。

数学层的容差取 5e-6 而不是机器精度：`WeldingCurve.__call__` 用分段线性插值，
两条焊接的采样锚点不重合时（源圆旋转后必然如此）必然留下 `O(h²·|f''|)` 的底噪，
加密采样不会消除它（错位误差只与 `h²` 有关，与 `T*` 精度无关）。
"""

from __future__ import annotations

import numpy as np
import pytest

import hbs
from hbs.conformal_welding import ConformalWelding, zipper_welding_prepared
from hbs._pipeline import prepare_boundary
from hbs.weld_compare import (
    WeldingCurve,
    analytic_welding,
    compare_conformal_welding,
    compare_welding,
    welding_curve,
)

_CIRCLE = 200
_DENSITY = 0.05

#: 非对称、无旋转对称的三齿形状（与 test_zipper_invariance 用同一族）。
_SHAPES = {
    "bent_lobe": lambda t: (1 + 0.35 * np.cos(2 * t - 0.6)) * (1 + 0.18 * np.cos(t + 1.1)),
    "asym_dent3": lambda t: 1 + 0.30 * np.cos(3 * t + 0.5) + 0.12 * np.cos(t - 0.9),
}


def _boundary(name: str, n: int = 500) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = _SHAPES[name](t)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _welding(name: str, method: str = "zipper") -> ConformalWelding:
    return hbs.get_hbs(_boundary(name), method=method, normalize="none",
                       circle_point_num=_CIRCLE)[2]


_ALPHA_A, _ALPHA_B = 0.25, 0.12


def _alpha(t):
    """解析相位增量：`0.25 sin(3θ+0.5) + 0.12 sin(θ-0.9) - α(0)`（保证 `f(0)=0`）。"""
    return (
        _ALPHA_A * np.sin(3 * np.asarray(t) + 0.5)
        + _ALPHA_B * np.sin(np.asarray(t) - 0.9)
        - (_ALPHA_A * np.sin(0.5) + _ALPHA_B * np.sin(-0.9))
    )


def _analytic() -> WeldingCurve:
    """解析焊接 `f(θ) = θ + α(θ)`。振幅满足 `3A + B = 0.87 < 1`，故严格单调。"""
    return analytic_welding(_alpha)


def _shift(curve: WeldingCurve, delta: float) -> WeldingCurve:
    """源圆旋转 `delta` 后的**同一条**焊接：`f₂(θ) = f₁(θ+δ) - f₁(δ)`。

    在解析相位上直接构造，不经过插值，因此与采样密度无关。
    """
    offset = _alpha(delta)
    return analytic_welding(lambda t: _alpha(np.asarray(t) + delta) - offset)


# --- 数学层：必须到机器精度 ---


def test_self_comparison_is_zero():
    """同一条焊接对自己：平移、规范、差距全为 0。"""
    curve = welding_curve(_welding("asym_dent3"))
    result = compare_welding(curve, curve)
    assert abs(result.shift) < 1e-6
    assert abs(result.gauge) < 1e-6
    assert result.rms_residual < 1e-9


@pytest.mark.parametrize("delta", [0.7, -2.1, 3.0, 1.9, 5.5])
def test_recovers_a_known_source_rotation(delta):
    """解析焊接上构造已知源圆旋转，必须到机器精度恢复，本质差距为 0。

    这是模块正确性的主要依据：它区分 `f(θ+T)-f(T)` 与 `f(θ-T)`。
    """
    curve = _analytic()
    result = compare_welding(curve, _shift(curve, delta))
    expected = np.mod(delta, 2 * np.pi)
    error = np.mod(result.shift - expected + np.pi, 2 * np.pi) - np.pi
    assert abs(error) < 1e-5, f"recovered {result.shift}, expected {expected}"
    assert result.rms_residual < 5e-6


def test_gauge_is_reported_separately_from_the_gap():
    """纯源圆旋转下，目标圆规范 `gauge` 与本质差距都不应被污染。"""
    curve = _analytic()
    result = compare_welding(curve, _shift(curve, 1.1))
    assert np.degrees(abs(result.gauge)) < 0.05
    assert result.rms_residual < 5e-6


def test_shift_sign_convention():
    """符号约定：`f₂(θ) = f₁(θ+δ) - f₁(δ)` 对应恢复出 `T* = +δ`。

    `f(θ-δ) - f(-δ)` 是同一个取值的另一种写法（它就是 δ 的绕向相反的源旋转），
    两者在圆上等价；这条测试把符号钉死，避免以后改动时静默翻转。
    """
    curve = _analytic()
    plus = compare_welding(curve, _shift(curve, 0.7))
    assert abs(np.mod(plus.shift - 0.7 + np.pi, 2 * np.pi) - np.pi) < 1e-5
    # 绕向相反的同一个旋转
    minus = analytic_welding(lambda t: _alpha(np.asarray(t) - 0.7) - _alpha(-0.7))
    back = compare_welding(curve, minus)
    assert abs(np.mod(back.shift + 0.7 + np.pi, 2 * np.pi) - np.pi) < 1e-5
    assert back.rms_residual < 1e-6


# --- 校验层 ---


def test_degree_one_is_enforced():
    """非次数 1 的目标角必须被拒绝，而不是静默给出无意义的数。"""
    curve = welding_curve(_welding("asym_dent3"))
    with pytest.raises(ValueError, match="degree-1"):
        WeldingCurve(
            source=curve.source.copy(),
            target=np.r_[curve.target[:-1] * 0.5, np.pi],
        )


def test_target_must_close_the_loop():
    """`target` 长度必须是 `len(source)+1`。"""
    curve = welding_curve(_welding("asym_dent3"))
    with pytest.raises(ValueError, match=r"len\(source\)\+1"):
        WeldingCurve(source=curve.source.copy(), target=curve.target[:-1])


def test_source_angles_must_be_distinct():
    """重复的源角说明焊接退化（`uniquify` 没生效），必须报错。"""
    cw = _welding("asym_dent3")
    cw.init_x_angle = np.zeros_like(cw.init_x_angle)
    cw.x = np.exp(1j * cw.init_x_angle)
    with pytest.raises(ValueError, match="distinct"):
        welding_curve(cw)


# --- 实现层：真实引擎，容差按离散误差 ---


def test_angular_and_zipper_welding_agree():
    """两条前向路径的焊接在源圆旋转下相同。

    这是 2.2.0 的关键契约。数值口径见 `runs/焊接对齐度量_2026-10-28/`：
    N=600 时本质差距约 0.07°，随 N 按 O(1/N) 收敛。
    """
    t = np.linspace(0, 2 * np.pi, 900, endpoint=False)
    r = _SHAPES["asym_dent3"](t)
    bound = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    prep = prepare_boundary(bound, 600)

    from hbs.angular import angular_welding
    from hbs.conformal_welding import ConformalWelding as CW

    zip_cw = zipper_welding_prepared(prep)
    ang_cw = CW.from_angular(angular_welding(prep, circle_num=600))
    result = compare_conformal_welding(zip_cw, ang_cw)

    assert np.degrees(result.rms_residual) < 0.5
    assert np.degrees(abs(result.gauge)) < 1.0


def test_conformal_welding_entry_point_matches_curve_entry():
    """两个入口给出的结果逐位相同。"""
    a = _welding("asym_dent3", method="zipper")
    b = _welding("bent_lobe", method="zipper")
    direct = compare_conformal_welding(a, b)
    via_curves = compare_welding(welding_curve(a), welding_curve(b))
    assert direct.shift == via_curves.shift
    assert direct.rms_residual == via_curves.rms_residual

