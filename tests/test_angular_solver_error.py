"""把角速度求解器自身的误差与下游（焊接插值、FEM 延拓）分开量。

`compute_direct_hbs(curve, derivative, pole=)` 吃解析曲线，`compute_direct_hbs_from_boundary`
吃点集并在归一化弧长上拟合周期三次样条。两条路用同一个 n、同一个极点，所以

- 两条路之差 = 曲线表示（解析 vs 样条）与配点分布的差异；
- 同一条路换 n 之差 = 求积误差。

两者都用 `interior_map` 在**固定物理点**上比较——它是 H 在点上的值，与边界参数化
无关。（点态比较 `alpha`/`beta` 没有意义：解析路线按 t 均匀配点，样条路线按弧长
均匀配点，两者相差 1.2e-1，属于重参数化，不是误差。）

2026-10-30 实测：

===========================  =============  ==========
量                             实测           阈值
===========================  =============  ==========
解析 vs 样条（n=1024）          5.23e-12       1e-10
n=256 vs n=1024                5.44e-12       1e-10
n=1024 vs n=8192               3.91e-13       （仅记录）
圆：max|alpha − t|              2.67e-15       1e-12
圆：密度常数性偏差              4.78e-13       1e-10
圆：|Möbius 中心|               2.31e-16       1e-12
===========================  =============  ==========

结论：**光滑边界上求解器本身是机器精度级准确的**，n=256 已收敛。所以库管线里
观察到的 1.23e-5（FEM 延拓 vs Poisson 积分，见 `test_angular_oracle.py`）与
2.6e-5（合成形状的保真度地板）**全部来自下游**：焊接的样条重参数化、
`linear_interp`、P1 延拓、μ 的面心离散与重建。它把「该优化哪一层」这个问题
从积分方程上移开了。
"""

from __future__ import annotations

import numpy as np
import pytest

from hbs._angular import direct

A, B = 1.3, 0.8


def _curve(t: np.ndarray) -> np.ndarray:
    return A * np.cos(t) + 1j * B * np.sin(t)


def _derivative(t: np.ndarray) -> np.ndarray:
    return -A * np.sin(t) + 1j * B * np.cos(t)


def _points(n: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([A * np.cos(t), B * np.sin(t)], axis=1)


def _queries() -> np.ndarray:
    """单位盘内的固定查询点（`interior_map` 的查询域是源单位盘）。"""
    k = np.arange(1, 25)
    return np.r_[0.0, 0.85 * np.exp(2j * np.pi * k / 24)]


def _solve_analytic(n: int = 1024):
    pts = _points(n)
    pole = direct._interior_pole(pts[:, 0] + 1j * pts[:, 1])
    return direct.compute_direct_hbs(_curve, _derivative, n=n, pole=pole)


def _relative(a: np.ndarray, ref: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.abs(a - ref) ** 2)) / np.sqrt(np.mean(np.abs(ref) ** 2))
    )


def test_analytic_curve_and_point_spline_agree():
    """1024 点上的样条表示与解析曲线等价到 1e-10（实测 5.2e-12）。"""
    analytic = _solve_analytic()
    spline = direct.compute_direct_hbs_from_boundary(_points(1024), n=1024)
    q = _queries()
    assert _relative(analytic.interior_map(q), spline.interior_map(q)) < 1e-10


def test_quadrature_is_converged_by_n_256():
    """n=256 与 n=1024 的差已在 1e-10 以内（实测 5.4e-12）。"""
    q = _queries()
    ref = direct.compute_direct_hbs_from_boundary(_points(1024), n=1024)
    coarse = direct.compute_direct_hbs_from_boundary(_points(1024), n=256)
    assert _relative(coarse.interior_map(q), ref.interior_map(q)) < 1e-10


def test_circle_solver_is_exact():
    """圆上真解是常数密度、α(t) = t：求解器必须复现到机器精度。"""
    n = 1024
    t = 2.0 * np.pi * np.arange(n) / n
    result = direct.compute_direct_hbs(
        lambda s: np.exp(1j * s),
        lambda s: 1j * np.exp(1j * s),
        n=n,
        pole=0j,
    )
    assert np.abs(result.alpha - t).max() < 1e-12
    rho = result.rho_internal
    assert (rho.max() - rho.min()) / rho.mean() < 1e-10
    assert abs(result.mobius_center) < 1e-12


def test_equations_are_solved_to_roundoff():
    """两条路的线性系统残差都该在 1e-13 以下（实测 1e-15 / 1e-14）。"""
    analytic = _solve_analytic()
    spline = direct.compute_direct_hbs_from_boundary(_points(1024), n=1024)
    for result in (analytic, spline):
        assert result.internal_equation_residual < 1e-13
        assert result.external_equation_residual < 1e-13


@pytest.mark.parametrize("n", [256, 512, 1024, 2048])
def test_densities_stay_positive(n):
    """解析椭圆上原始内外密度严格为正，无需密度修剪。"""
    result = _solve_analytic(n)
    assert result.min_raw_internal_density > 0
    assert result.min_raw_external_density > 0
    assert result.density_regularization_mass == 0.0
