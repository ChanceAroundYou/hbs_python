"""HBS round-trip + 方向不变回归测试（锁定前向修复 + 重建尺度定律 1/rx²）。"""
import numpy as np
import pytest

from hbs import get_hbs, reconstruct_from_hbs
from hbs.mesh import get_unit_disk


def _shoelace_area(z):
    z = np.asarray(z)
    if z.ndim == 2:
        z = z[:, 0] + 1j * z[:, 1]
    zc = np.concatenate([z, z[:1]])
    return 0.5 * np.abs(np.imag(np.conj(zc[:-1]) * zc[1:])).sum()


def _roundtrip_area_ratio(bound, disk, clockwise=True):
    b = bound if clockwise else bound[::-1]
    hbs, _, _, _ = get_hbs(b, 1000, 0.01, disk)
    rec, _, _, _ = reconstruct_from_hbs(hbs, disk)
    a0 = _shoelace_area(bound)
    a1 = _shoelace_area(rec)
    return a1 / a0 if a0 > 0 else np.nan


def _ellipse(rx=1.0, ry=0.5, n=500):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([rx * np.cos(t), ry * np.sin(t)], 1)


# --- 方向不变（前向修复 1.1）---

def test_orientation_invariance(disk):
    ccw = _ellipse(1.0, 0.5)
    h_ccw, _, _, _ = get_hbs(ccw, 1000, 0.01, disk)          # 逆时针（内部自动顺）
    h_cw, _, _, _ = get_hbs(ccw[::-1], 1000, 0.01, disk)     # 顺时针
    assert np.allclose(h_ccw, h_cw, atol=1e-6)


# --- round-trip 面积比（健康的中等形变区间）---

def test_thin_ellipse_roundtrip_area(disk):
    # ry=0.5 的中等形变椭圆在焊接数值误差内面积比≈1（已验证 300/500/800 全部健康）
    ratio = _roundtrip_area_ratio(_ellipse(1.0, 0.5), disk)
    assert np.isfinite(ratio)
    assert abs(ratio - 1.0) < 0.2


def test_circle_roundtrip_area(disk):
    ratio = _roundtrip_area_ratio(_ellipse(1.0, 1.0), disk)
    assert np.isfinite(ratio)
    assert abs(ratio - 1.0) < 0.2


def test_roundtrip_area_scales_with_landmark(disk):
    # μ 尺度不变：重建尺度由 landmark（disk 的 (1,0)→target(1,0)）决定，强制形状 x 半轴
    # →1.0，面积比 = 1/rx²（形状本身保真，纯均匀缩放 rec×rx≈orig）。非 seam 退化、
    # 非近圆病态——是 μ 表示法的数学事实。锁死该定律，landmark 逻辑变动会在此捕获。
    rx = 1.2
    ratio = _roundtrip_area_ratio(_ellipse(rx, 1.0), disk)
    assert np.isfinite(ratio)
    assert abs(ratio - 1 / rx**2) < 0.2


# --- 非椭圆光滑形状 round-trip（尺度无关形状保真）---

def _superellipse(a=1.2, b=1.0, n=4.0, m=500):
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    x = a * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / n)
    y = b * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / n)
    return np.stack([x, y], 1)


def _blob(m=500):
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    r = 1 + 0.2 * np.cos(3 * t)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def _peanut(m=500):
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    x = (1 + 0.3 * np.cos(2 * t)) * np.cos(t)
    y = (0.8 + 0.2 * np.sin(2 * t)) * np.sin(t)
    return np.stack([x, y], 1)


def _resample(z, n=400):
    z = np.asarray(z, complex)
    z = np.concatenate([z, z[:1]])
    d = np.abs(np.diff(z))
    s = np.concatenate([[0], np.cumsum(d)])
    s = s / s[-1]
    return np.interp(np.linspace(0, 1, n, endpoint=False), s, z, period=1.0)


def _unit_area_fidelity(shape, rec):
    """尺度无关形状保真：两边都中心化 + 归一化到单位面积 + 循环对齐取最小平均距离。"""
    o = _resample(shape[:, 0] + 1j * shape[:, 1])
    r = _resample(rec[:, 0] + 1j * rec[:, 1])
    o = o - o.mean()
    r = r - r.mean()
    o = o / np.sqrt(_shoelace_area(o))
    r = r / np.sqrt(_shoelace_area(r))
    return min(np.mean(np.abs(np.roll(r, k) - o)) for k in range(0, 400, 20))


@pytest.mark.parametrize(
    "shape_fn,name", [(_superellipse, "superellipse"), (_blob, "blob"), (_peanut, "peanut")]
)
def test_non_ellipse_roundtrip_shape_fidelity(disk, shape_fn, name):
    shape = shape_fn()
    hbs, _, _, d = get_hbs(shape, 1000, 0.01, disk)
    rec, _, _, _ = reconstruct_from_hbs(hbs, d)
    f = _unit_area_fidelity(shape, rec)
    assert f < 0.15, f"{name} 尺度无关形状保真 {f:.3f} 超出容差"


# --- 输入校验 / 行为边界 ---

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sharp_corner_square_raises(disk):
    # 尖角（正方形）在 zipper Möbius 定心不收敛 → RuntimeError（快速失败而非静默坏输出）
    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    square = np.stack([np.sign(np.cos(t)), np.sign(np.sin(t))], 1)
    with pytest.raises(RuntimeError):
        get_hbs(square, 1000, 0.01, disk)


def test_reconstruct_wrong_disk_asserts():
    d1 = get_unit_disk(0.01, 500)
    d2 = get_unit_disk(0.01, 300)
    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    ell = np.stack([1.2 * np.cos(t), np.sin(t)], 1)
    hbs, _, _, _ = get_hbs(ell, 500, 0.01, d1)
    with pytest.raises(AssertionError):
        reconstruct_from_hbs(hbs, d2)


def test_get_hbs_auto_disk_roundtrip():
    # 不传 disk → 自动建（circle_point_num=500），返回的 disk 可用于重建
    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    ell = np.stack([1.2 * np.cos(t), np.sin(t)], 1)
    hbs, _, _, disk = get_hbs(ell, 500, 0.01)
    assert disk.circle_num == 500
    rec, _, _, _ = reconstruct_from_hbs(hbs, disk)
    assert np.all(np.isfinite(rec))


@pytest.mark.slow
def test_many_boundaries_healthy_regime(disk):
    """分辨率/长宽比冒烟：中等形变区间 300/500/800 全不崩且面积比合理。"""
    for n in (300, 500, 800):
        for rx, ry in ((1.0, 0.5), (1.0, 1.0)):
            ratio = _roundtrip_area_ratio(_ellipse(rx, ry, n), disk)
            assert np.isfinite(ratio), f"n={n} rx={rx} ry={ry} NaN"
            assert abs(ratio - 1.0) < 0.5
