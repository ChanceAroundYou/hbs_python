"""HBS round-trip + 方向不变回归测试（锁定前向修复 + 重建尺度定律 1/rx²）。"""
import numpy as np
import pytest

from hbs import get_hbs, reconstruct_from_hbs


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


@pytest.mark.slow
def test_many_boundaries_healthy_regime(disk):
    """分辨率/长宽比冒烟：中等形变区间 300/500/800 全不崩且面积比合理。"""
    for n in (300, 500, 800):
        for rx, ry in ((1.0, 0.5), (1.0, 1.0)):
            ratio = _roundtrip_area_ratio(_ellipse(rx, ry, n), disk)
            assert np.isfinite(ratio), f"n={n} rx={rx} ry={ry} NaN"
            assert abs(ratio - 1.0) < 0.5
