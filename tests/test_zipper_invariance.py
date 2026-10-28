"""zipper 前向的**彻底不变性**契约（2.1.0）。

2.0.x 的 zipper 有两处破坏不变性的来源，本文件把修复钉成可执行契约：

1. 8 轮 `rotate_x` → `linear_interp` → 重算场的 λ 归一化。`rotate_x(r)` 转的
   是源圆参数化，紧跟的 `linear_interp` 总是重采样到**从角 0 起的固定 x 网格**；
   r 不是网格步长整数倍时这不是刚体旋转。停止门 `|r| <= 5e-3` 留下的残余随输入
   起点变化，seam 处 dμ/dr 增益实测 ~125×，于是 ~0.6 量级的 |μ| 差异被写进结果。
   修法：改成一次性 I₂ 规范（`μ ← e^{-2iθ}μ` 且顶点乘 `e^{-iθ}`），不重解。

2. `_resample_boundary` 把弧长网格原点钉在 `bound[0]` 上，输入起点一挪，重采样
   的**点集本身**就变了。修法：先定向、再按几何锚点 roll、最后重采样。

测试用**非对称、良态**的形状（无旋转对称、max|μ| < 1）。对称形状有并列尖点，
锚点天然多值，不属于本契约的适用范围。
"""

import numpy as np
import pytest

import hbs
import hbs.hbs as hbs_mod
from hbs._pipeline import prepare_boundary
from hbs.conformal_welding import get_conformal_welding
from hbs.mesh import get_unit_disk

#: 非对称、无旋转对称、max|μ| < 0.7 的两族参数曲线。
_SHAPES = {
    "bent_lobe": lambda t: (1 + 0.35 * np.cos(2 * t - 0.6))
    * (1 + 0.18 * np.cos(t + 1.1)),
    "asym_dent3": lambda t: 1
    + 0.30 * np.cos(3 * t + 0.5)
    + 0.12 * np.cos(t - 0.9),
}

_CIRCLE = 200
_DENSITY = 0.05


def _boundary(name, n=500):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = _SHAPES[name](t)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _rotate(b, a):
    c, s = np.cos(a), np.sin(a)
    return b @ np.array([[c, s], [-s, c]])


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


@pytest.fixture(scope="module")
def disk():
    """Invariance lives in the welding solve, not the FEM disk — keep it small.

    `circle_point_num` 必须与盘一致，否则 `cw.linear_interp` 与盘的边界点数不符。
    """
    return get_unit_disk(_DENSITY, _CIRCLE)


def _mu(bound, disk):
    return hbs.get_hbs(
        bound, method="zipper", normalize="i2",
        disk=disk, circle_point_num=_CIRCLE,
    )[0]


@pytest.mark.parametrize("name", list(_SHAPES))
@pytest.mark.parametrize("shift", [1, 37, 250])
def test_start_point_roll_is_bitwise_invariant(disk, name, shift):
    """起点沿曲线平移 k 个点，μ 场逐位不变（不是"接近"）。

    这是 `_prepare_boundary` 里几何锚点的直接后果：锚点只依赖点集，roll 变成了
    纯循环移位，重采样的点集与顺序完全一致。修复前同一测试的残差是 0.10–2.20。
    """
    b = _boundary(name)
    assert np.array_equal(_mu(np.roll(b, shift, axis=0), disk), _mu(b, disk))


@pytest.mark.parametrize("name", list(_SHAPES))
def test_reversal_is_bitwise_invariant(disk, name):
    """反转绕行方向（同一形状的两种采样）落到同一锚点与同一 μ。"""
    b = _boundary(name)
    assert np.array_equal(_mu(b[::-1], disk), _mu(b, disk))


@pytest.mark.parametrize("name", list(_SHAPES))
def test_rotation_scale_translation_to_fp_precision(disk, name):
    """旋转 / 缩放 / 平移下 μ 只剩浮点噪声（< 1e-8 相对）。

    zipper 的焊接对应 `(x, y)` 本身与形状的刚体变换无关（实测差 ~1e-13），所以
    这里在**固定盘**上比较 —— 与角速度路径不同，不需要同步旋转网格。
    """
    b = _boundary(name)
    mu0 = _mu(b, disk)
    assert _rel(_mu(_rotate(b, 0.7), disk), mu0) < 1e-8
    assert _rel(_mu(b * 3.7, disk), mu0) < 1e-8
    assert _rel(_mu(b + np.array([2.3, -4.1]), disk), mu0) < 1e-8


def test_gauge_no_longer_iterates():
    """旧的 8 轮 λ 循环必须消失（"旧方式验证完成后删掉"）。"""
    assert not hasattr(hbs_mod, "_LAMBDA_MAX_ITERATIONS")
    assert not hasattr(hbs_mod, "_lambda_phase")


def test_zipper_gauge_preserves_magnitude(disk):
    """I₂ 是一次共轭旋转，只改相位：|μ| 与未规范化的场相同（到浮点舍入）。"""
    b = _boundary("bent_lobe", n=200)
    kw = dict(method="zipper", disk=disk, circle_point_num=_CIRCLE)
    mu_raw = hbs.get_hbs(b, normalize="none", **kw)[0]
    mu_i2 = hbs.get_hbs(b, normalize="i2", **kw)[0]
    assert np.allclose(np.abs(mu_i2), np.abs(mu_raw), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", list(_SHAPES))
def test_two_public_entries_agree(name):
    """`get_hbs` 与 `get_conformal_welding(method="zipper")` 共用一份预处理。

    两者若在预处理上分叉，同一形状会给出不同的 `(x, y)`（修复前 roll 后排序 x
    相差 5.56），下游无法互相校验。比较 `normalize="none"` 的契约：I₂ 规范会
    额外旋转源角，那是 `get_hbs` 后处理的一步，不在焊接求解里。
    """
    b = _boundary(name)
    # 2.2.0：两条路径的焊接都来自角速度解（`angular_welding`），所以同一
    # `normalize` 下 `(x, y)` 逐位相同。`get_conformal_welding` 的默认
    # `normalize` 是 "i2"（与 `get_hbs` 一致），这里显式对齐到 "none" 比较
    # 求解本身的输出。
    for norm in ("none", "i2"):
        cw_hbs = hbs.get_hbs(b, method="zipper", normalize=norm)[2]
        cw_direct = get_conformal_welding(b, method="zipper", normalize=norm)
        # 逐元素对拍须在圆上做差：`mod(·, 2π)` 会把 0 与 2π 的同一角分成两个值，
        # `array_equal` 会误报。
        dx = np.mod(np.angle(cw_hbs.init_x) - np.angle(cw_direct.init_x) + np.pi,
                    2 * np.pi) - np.pi
        dy = np.mod(np.angle(cw_hbs.init_y) - np.angle(cw_direct.init_y) + np.pi,
                    2 * np.pi) - np.pi
        assert np.abs(dx).max() < 1e-12, f"{norm}: x 不一致 {np.abs(dx).max():.2e}"
        assert np.abs(dy).max() < 1e-12, f"{norm}: y 不一致 {np.abs(dy).max():.2e}"


def test_prepare_boundary_is_the_roll_equivariant_step():
    """锚点只依赖点集：roll 后预处理逐位相同；旋转/缩放协变。

    注意 roll 之后**索引**当然会变（同一几何点落到另一个下标），所以这里比较
    的是点集层面的结果，不是 `_canonical_start_index` 的返回值。
    """
    b = _boundary("asym_dent3")
    p0 = prepare_boundary(b, 500)
    for k in (1, 37, 250):
        rolled = np.roll(b, k, axis=0)
        assert np.array_equal(prepare_boundary(rolled, 500), p0)
        # 锚点是同一个几何点，只是下标平移了 k
        i0 = hbs_mod._canonical_start_index(b)
        ik = hbs_mod._canonical_start_index(rolled)
        assert np.allclose(rolled[ik], b[i0])
        assert ik == (i0 + k) % len(b)

    a = 0.7
    assert np.allclose(prepare_boundary(_rotate(b, a), 500), _rotate(p0, a))
    assert np.allclose(prepare_boundary(b * 3.7, 500), 3.7 * p0)
    assert np.array_equal(prepare_boundary(b[::-1], 500), p0)
