"""功能测试：合成图像 → get_boundary → get_hbs → reconstruct 端到端。

覆盖 README 主链路（图像 → 边界 → HBS → 重建）。
注意坐标约定：get_boundary 输出图像坐标（y 向下），重建输出数学坐标（y 向上），
因此先把边界 y 翻到数学坐标再做 round-trip（否则重建是原图的 y 镜像）。
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # noqa: F401
from hbs import get_hbs, reconstruct_from_hbs  # noqa: E402
from hbs.utils.boundary import get_boundary  # noqa: E402


def _shoelace_area(z):
    z = np.asarray(z)
    if z.ndim == 2:
        z = z[:, 0] + 1j * z[:, 1]
    zc = np.concatenate([z, z[:1]])
    return 0.5 * np.abs(np.imag(np.conj(zc[:-1]) * zc[1:])).sum()


def _resample(z, n=300):
    z = np.asarray(z, complex)
    z = np.concatenate([z, z[:1]])
    d = np.abs(np.diff(z))
    s = np.concatenate([[0], np.cumsum(d)])
    s = s / s[-1]
    return np.interp(np.linspace(0, 1, n, endpoint=False), s, z, period=1.0)


def test_image_to_hbs_roundtrip_shape_fidelity(tmp_path):
    img_path = str(tmp_path / "ellipse.png")
    img = np.full((200, 200), 255, np.uint8)
    cv2.ellipse(img, (100, 100), (70, 40), 0, 0, 360, 0, -1)
    cv2.imwrite(img_path, img)

    bound = get_boundary(img_path, num_points=250)
    assert bound.shape == (250, 2)
    assert np.isfinite(bound).all()

    # 图像坐标 y 向下 → 数学坐标 y 向上（镜像约定）
    bound_math = bound.copy()
    bound_math[:, 1] = -bound_math[:, 1]

    hbs, _, _, disk = get_hbs(bound_math, 500, 0.01)
    rec, _, _, _ = reconstruct_from_hbs(hbs, disk)

    o = _resample(bound_math[:, 0] + 1j * bound_math[:, 1])
    r = _resample(rec[:, 0] + 1j * rec[:, 1])
    o = o - o.mean()
    r = r - r.mean()
    o = o / np.sqrt(_shoelace_area(o))
    r = r / np.sqrt(_shoelace_area(r))
    best = min(np.mean(np.abs(np.roll(r, k) - o)) for k in range(0, 300, 15))
    # 像素化边界（200px 图）的容差放宽到 0.25（λ 归一化后重建略变）
    assert best < 0.25
