"""无网格参照：给焊接插值与 FEM 延拓定绝对误差界。

`DirectHBS` 自带两条**不经过**库管线离散化的路径：

- `boundary_map(theta)`：直接在源角上给出 H 的边界值，不经过
  `ConformalWelding.linear_interp` 的插值与目标圆相位归零；
- `interior_map(z)`：边界层单层势的 Poisson 积分，无网格、无 P1 延拓。

拿它们当参照，就能把「焊接插值错多少」与「FEM 延拓错多少」分开量，而不是只做
管线自身的往返比较。基准值来自 2026-10-30 的实测（每例的阈值取实测的 2–3 倍）：

===========================  ==============  ==============
用例                          实测             阈值
===========================  ==============  ==============
圆：boundary_map vs init_y    9.55e-16        1e-12
圆：interior_map vs FEM rms   6.01e-11        1e-8
椭圆：boundary_map vs init_y  8.38e-6         2e-5
椭圆：interior_map vs FEM rms 1.23e-5         3e-5
===========================  ==============  ==============

椭圆上的偏差不是 bug：两条路径用不同的插值方案（`boundary_map` 在 β→α 上做
`np.interp`；焊接用 `CubicSpline(α−t)` + `Pchip(β→t)`），8.4e-6 就是这套差异
的量级，与合成形状 2.6e-5 的保真度地板同一量级。
"""

from __future__ import annotations

from functools import cache

import numpy as np

from hbs._angular import direct
from hbs._pipeline import build_hbs, prepare_boundary, resolve_disk
from hbs.angular import angular_welding
from hbs.conformal_welding import ConformalWelding

CIRCLE_NUM = 500
SOLVER_N = 1024
POINTS = 1024


def _boundary(kind: str, n: int = POINTS) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if kind == "circle":
        return np.stack([np.cos(t), np.sin(t)], axis=1)
    return np.stack([1.3 * np.cos(t), 0.8 * np.sin(t)], axis=1)


@cache
def _chain(kind: str):
    """bound → DirectHBS → 焊接 → 盘网格 → FEM 场（`normalize="none"`）。"""
    bound = _boundary(kind)
    prepared = prepare_boundary(bound, CIRCLE_NUM, resample=False)
    solver = direct.compute_direct_hbs_from_boundary(prepared, n=SOLVER_N)
    welding = ConformalWelding.from_angular(
        angular_welding(prepared, circle_num=CIRCLE_NUM, n=SOLVER_N)
    )
    # 引擎侧的取样步骤（`_engines._result` 做的事）：管线本身不再插值。
    welding.linear_interp(CIRCLE_NUM)
    disk, fem = resolve_disk(None, 0.01, CIRCLE_NUM)
    _mu, field, welding, disk, _gauge = build_hbs(
        np.asarray(welding.y, complex), welding, disk, fem, "none"
    )
    return prepared, solver, welding, disk, field


def _complex(field: np.ndarray) -> np.ndarray:
    return field[:, 0] + 1j * field[:, 1]


def _interior(disk, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """盘网格的内点（不含源圆顶点）与对应的 FEM 场值。"""
    z = disk.in_vert[:, 0] + 1j * disk.in_vert[:, 1]
    start = disk.circle_num
    return z, _complex(field)[start : start + disk.in_vert_num]


def test_circle_boundary_map_matches_welding():
    """圆上两条边界路径必须一致到机器精度（无插值差异可归咎）。"""
    _prepared, solver, welding, _disk, _field = _chain("circle")
    theta = np.asarray(welding.init_x_angle, float)
    assert np.abs(solver.boundary_map(theta) - welding.init_y).max() < 1e-12


def test_circle_interior_map_matches_fem():
    """FEM 延拓在圆上应复现 Poisson 积分（P1 对线性函数精确）。"""
    _prepared, solver, _welding, disk, field = _chain("circle")
    z, fem_in = _interior(disk, field)
    exact = solver.interior_map(z)
    rms = np.sqrt(np.mean(np.abs(fem_in - exact) ** 2)) / np.sqrt(
        np.mean(np.abs(fem_in) ** 2)
    )
    assert rms < 1e-8


def test_circle_fem_field_is_a_rotation_of_identity():
    """真解是 H(z) = c·z（|c| = 1）：FEM 场与它必须只差一个整体旋转。"""
    _prepared, _solver, _welding, disk, field = _chain("circle")
    z, fem_in = _interior(disk, field)
    c = np.vdot(z, fem_in) / np.vdot(z, z)
    assert abs(abs(c) - 1.0) < 1e-8
    resid = np.abs(fem_in - c * z)
    assert resid.max() / np.sqrt(np.mean(np.abs(fem_in) ** 2)) < 1e-7


def test_ellipse_boundary_map_matches_welding():
    """椭圆上的差是两套插值方案之差，2026-10-30 实测 8.38e-6。"""
    _prepared, solver, welding, _disk, _field = _chain("ellipse")
    theta = np.asarray(welding.init_x_angle, float)
    assert np.abs(solver.boundary_map(theta) - welding.init_y).max() < 2e-5


def test_ellipse_interior_map_matches_fem():
    """椭圆上 FEM 延拓相对 Poisson 积分的误差，实测 rms 1.23e-5。"""
    _prepared, solver, _welding, disk, field = _chain("ellipse")
    z, fem_in = _interior(disk, field)
    exact = solver.interior_map(z)
    rms = np.sqrt(np.mean(np.abs(fem_in - exact) ** 2)) / np.sqrt(
        np.mean(np.abs(fem_in) ** 2)
    )
    assert rms < 3e-5
