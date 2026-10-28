"""前向不变性契约：两个引擎 × 刚体变换 / 起始点 / 采集密度。

与 `test_zipper_invariance.py` 的分工：那个文件钉 zipper 的历史修复（8 轮 λ 循环、
弧长锚点），只测 zipper 且只用 `normalize="i2"`。这里测**当前**的公开契约：

1. 旋转 / 平移 / 缩放后焊接与 μ 不变 —— 两个引擎都要，且要覆盖复合变换；
2. 起始点循环移位后 μ **逐位**相同；
3. 采集密度是**收敛量**不是不变量：加密到细档时应单调逼近。

**不**在这里钉"某个形状在某个网格上的收敛速率"：那是研究观测，依赖具体的
`circle_point_num` / `density` 配置，写在库的测试里会在换配置时变成假阳性。
梨 / zipper 在 `CIRCLE=500` 上的边界层收敛停滞记录在
`HBS-research/runs/前向不变性校验_2026-10-28/报告.md` §3。

GPU 无关、无随机数，全部确定性。
"""

from __future__ import annotations

import numpy as np
import pytest

import hbs
from hbs._pipeline import prepare_boundary
from hbs.conformal_welding import ConformalWelding, zipper_welding_prepared
from hbs.mesh import get_unit_disk
from hbs.weld_compare import compare_conformal_welding

import hbs.angular as angular

CIRCLE = 120
DENSITY = 0.05
WELD_N = 400
TP = 2 * np.pi

_SHAPES = {
    "bent_lobe": lambda t: (1 + 0.30 * np.cos(t - 1.0)) * np.exp(1j * t),
    "-": lambda t: (1 + 0.22 * np.cos(t + 0.6) + 0.12 * np.cos(2 * t - 1.4)) * np.exp(1j * t),
}
for _name in ("bent_lobe",):
    _SHAPES["asym_dent3"] = lambda t: (1 + 0.30 * np.cos(3 * t + 0.5)
                                       + 0.12 * np.cos(t - 0.9)) * np.exp(1j * t)


def _boundary(zfun, n=WELD_N):
    t = np.linspace(0, TP, n, endpoint=False)
    z = zfun(t)
    return np.stack([z.real, z.imag], axis=1)


def _rotate(b, angle):
    c, s = np.cos(angle), np.sin(angle)
    return b @ np.array([[c, s], [-s, c]])


def _weld(prepared, engine):
    if engine == "angular":
        return ConformalWelding.from_angular(
            angular.angular_welding(prepared, circle_num=WELD_N)
        )
    return zipper_welding_prepared(prepared)


def _mu(bound, engine, disk):
    return hbs.get_hbs(bound, method=engine, disk=disk,
                       circle_point_num=CIRCLE)[0]


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


@pytest.fixture(scope="module")
def disk():
    return get_unit_disk(DENSITY, CIRCLE)


# --- 1. 刚体变换 ---

_TRANSFORMS = {
    "rotate_0.7": lambda b: _rotate(b, 0.7),
    "rotate_2.3": lambda b: _rotate(b, 2.3),
    "translate": lambda b: b + np.array([2.3, -4.1]),
    "scale_3.7": lambda b: b * 3.7,
    "scale_0.13": lambda b: b * 0.13,
    "compose": lambda b: _rotate(b * 2.4, 1.1) + np.array([-3.0, 5.5]),
}


@pytest.mark.parametrize("engine", ["angular", "zipper"])
@pytest.mark.parametrize("name", list(_SHAPES))
def test_rigid_transform_invariance(disk, engine, name):
    """旋转 / 平移 / 缩放（含复合）后 μ 只剩浮点噪声。

    阈值 1e-6 相对：实测 angular 是 1e-11、zipper 是 1e-9，留足两个量级余量。
    这条同时覆盖"两个引擎都协变"——只测一个会漏掉某条路径单独退化。
    """
    b = _boundary(_SHAPES[name])
    base = _mu(b, engine, disk)
    for label, fn in _TRANSFORMS.items():
        assert _rel(_mu(fn(b), engine, disk), base) < 1e-6, f"{engine}/{label}"


@pytest.mark.parametrize("engine", ["angular", "zipper"])
def test_rigid_transform_leaves_the_welding_alone(engine):
    """焊接 `(x, y)` 本身在刚体变换下不变（不是靠规范吸收）。"""
    b = _boundary(_SHAPES["asym_dent3"])
    base = _weld(prepare_boundary(b, WELD_N), engine)
    for fn in (_TRANSFORMS["rotate_0.7"], _TRANSFORMS["compose"]):
        cw = _weld(prepare_boundary(fn(b), WELD_N), engine)
        rms = np.degrees(compare_conformal_welding(base, cw).rms_residual)
        assert rms < 1e-6, f"{engine}: 焊接变了 {rms:.2e}°"


# --- 2. 起始点 ---

@pytest.mark.parametrize("engine", ["angular", "zipper"])
@pytest.mark.parametrize("shift", [1, 37, 211])
def test_start_point_roll_is_bitwise_invariant(disk, engine, shift):
    """循环移位起点后 μ **逐位**相同（几何锚点只依赖点集）。"""
    b = _boundary(_SHAPES["bent_lobe"])
    assert np.array_equal(_mu(np.roll(b, shift, axis=0), engine, disk),
                          _mu(b, engine, disk))


# --- 3. 采集密度是收敛量 ---

@pytest.mark.parametrize("engine", ["angular", "zipper"])
def test_density_refinement_converges(disk, engine):
    """加密输入采样时，μ 单调逼近最细档（相对差随 n 下降）。

    不要求逐位不变——采样点数是离散参数，不同密度本来给不同的离散焊接。
    """
    zfun = _SHAPES["asym_dent3"]
    coarse = _mu(_boundary(zfun, 200), engine, disk)
    mid = _mu(_boundary(zfun, 800), engine, disk)
    fine = _mu(_boundary(zfun, 3000), engine, disk)
    d_mid = _rel(mid, fine)
    d_coarse = _rel(coarse, fine)
    assert d_mid < d_coarse, (
        f"{engine}: 加密没有降低差异 ({d_coarse:.3e} -> {d_mid:.3e})"
    )
