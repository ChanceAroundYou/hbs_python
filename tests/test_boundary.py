"""boundary 工具测试：smooth_resample / anti_aliasing / extract_boundary_points。

boundary 模块顶层 import cv2，因此无 opencv 环境整个文件跳过。
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # noqa: F401  # boundary 模块硬依赖 cv2，缺失则整文件跳过
from hbs.utils.boundary import (  # noqa: E402
    anti_aliasing,
    extract_boundary_points,
    smooth_resample,
)


def _circle(n=200, radius=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(t), radius * np.sin(t)], 1)


# --- smooth_resample（纯 numpy/scipy，无 cv2 依赖）---

def test_smooth_resample_output_shape():
    out = smooth_resample(_circle(50), num_points=200)
    assert out.shape == (200, 2)
    assert np.all(np.isfinite(out))


def test_smooth_resample_uniform_arc_length():
    out = smooth_resample(_circle(50), num_points=100)
    d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(d, d.mean(), rtol=1e-4)


def test_smooth_resample_handles_open_boundary():
    # 开链自动闭合；返回 num_points 个点覆盖一整圈（末点即回起点前一格）
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    out = smooth_resample(sq, num_points=8)
    assert out.shape == (8, 2)
    d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    wrap = np.linalg.norm(out[0] - out[-1])  # 首尾隐式闭合一步
    assert np.allclose(d, d.mean(), rtol=1e-4)
    assert np.isclose(wrap, d.mean(), rtol=1e-4)


def test_smooth_resample_preserves_shape():
    out = smooth_resample(_circle(200), num_points=100)
    # 100 边形弦中点半径略 <1（sagitta），容差放 0.01
    assert np.abs(np.linalg.norm(out, axis=1) - 1.0).max() < 0.01


# --- anti_aliasing（cv2）---

def test_anti_aliasing_shape_and_finite():
    rng = np.random.default_rng(0)
    pts = _circle(100) + rng.normal(0, 0.1, (100, 2))
    out = anti_aliasing(pts, 5)
    assert out.shape == pts.shape
    assert np.all(np.isfinite(out))


def test_anti_aliasing_actually_smooths():
    # 平滑降噪：含噪圆经 anti_aliasing 后总变差显著下降（回归：cv2 5.0 下曾是 no-op）
    rng = np.random.default_rng(0)
    noisy = _circle(100) + rng.normal(0, 0.1, (100, 2))
    sm = anti_aliasing(noisy, 9)
    tv_in = np.abs(np.diff(noisy, axis=0)).sum()
    tv_out = np.abs(np.diff(sm, axis=0)).sum()
    assert tv_out < tv_in * 0.9


# --- extract_boundary_points（cv2，合成图像）---

def _write_square_image(path):
    img = np.full((200, 200), 255, np.uint8)
    img[60:141, 60:141] = 0  # 80x80 黑方块
    cv2.imwrite(path, img)


def test_extract_boundary_points_on_square(tmp_path):
    img_path = str(tmp_path / "square.png")
    _write_square_image(img_path)
    pts = extract_boundary_points(img_path)
    assert pts.shape[1] == 2
    assert len(pts) > 10
    # 轮廓点落在方块边界（60..140，含 padding 偏移 1）
    assert pts[:, 0].min() >= 59
    assert pts[:, 0].max() <= 141
    assert pts[:, 1].min() >= 59
    assert pts[:, 1].max() <= 141
