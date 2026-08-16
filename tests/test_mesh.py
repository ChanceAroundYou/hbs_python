"""Mesh/DiskMesh 基础算子测试。"""
import numpy as np
import pytest

from hbs.mesh import Mesh, get_rect, get_unit_disk, get_unit_disk_in_rect


@pytest.fixture
def m():
    return get_rect(width=1, height=1, density=0.25)


def test_rect_mesh_basics():
    m = get_rect(width=1, height=1, density=0.5)
    # 3x3 grid -> 9 vert, Delaunay triangularization
    assert m.vert_num == 9
    assert m.face.shape[1] == 3
    assert m.face_center.shape == (m.face_num, 2)
    assert np.isfinite(m.area).all()
    # 微分算子形状：face_num x vert_num
    assert m.Dx.shape == (m.face_num, m.vert_num)
    assert m.Dy.shape == (m.face_num, m.vert_num)


def test_unit_disk_structure(disk):
    assert disk.circle_num == 1000
    assert disk.in_vert_num > 0
    # face 中心在单位圆盘内
    assert np.linalg.norm(disk.face_center, axis=1).max() <= 1.0 + 1e-9


def test_gradient_operator_convention():
    # 面内常梯度：Dx/Dy 对坐标函数给出面内梯度（含顺时针取向符号）。
    # 实测 Dx@x=-1, Dy@y=-1, 交叉 0 —— 锁定该约定（与 HBS 顺时针边界一致）。
    m = get_rect(width=1, height=1, density=0.25)
    x = m.vert[:, 0]
    y = m.vert[:, 1]
    assert np.allclose(np.asarray(m.Dx @ x).ravel(), -1.0, atol=1e-8)
    assert np.allclose(np.asarray(m.Dy @ y).ravel(), -1.0, atol=1e-8)
    assert np.allclose(np.asarray(m.Dx @ y).ravel(), 0.0, atol=1e-8)
    assert np.allclose(np.asarray(m.Dy @ x).ravel(), 0.0, atol=1e-8)


def test_laplacian_shape_and_constant_kernel(m):
    assert m.laplacian.shape == (m.vert_num, m.vert_num)
    # 常函数在 Laplacian 核里（离散调和）应近似 0
    ones = np.ones(m.vert_num)
    lap = np.asarray(m.laplacian @ ones).ravel()
    assert np.allclose(lap, 0.0, atol=1e-8)


def test_get_unit_disk_in_rect():
    base = get_unit_disk(0.05, 100)
    inr = get_unit_disk_in_rect(base, height=4, width=4, density=0.05)
    # 圆盘部分保留，矩形外圈顶点追加
    assert inr.circle_num == 100
    assert inr.in_vert_num == base.in_vert_num
    assert inr.out_vert_num > 0
    assert inr.face_num > base.face_num
    # 全部顶点在矩形内（半宽/半高 = 2）；out_vert 全在单位圆外
    assert np.abs(inr.vert).max() <= 2.0 + 1e-9
    assert np.linalg.norm(inr.out_vert, axis=1).min() > 1.0
