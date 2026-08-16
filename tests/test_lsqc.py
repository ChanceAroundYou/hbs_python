"""LSQC 求解器测试。"""
import numpy as np
import pytest

from hbs.mesh import get_rect
from hbs.qc.lsqc import lsqc_solver


@pytest.fixture
def mesh():
    return get_rect(width=1, height=1, density=0.05)


def test_landmark_pins_vertices(mesh):
    # 选定两个 landmark -> (0,0) 和 (1,0)，μ=0 时其余自由但 landmark 必须钉住
    mu = np.zeros(mesh.face_num, dtype=complex)
    target = np.array([[0.0, 0.0], [0.8, 0.0]])
    # 选距 (0,0) 与 (1,0) 最近的顶点
    dist0 = np.linalg.norm(mesh.vert, axis=1)
    dist1 = np.linalg.norm(mesh.vert - [1, 0], axis=1)
    lm = np.array([dist0.argmin(), dist1.argmin()])
    sol = lsqc_solver(mu, lm, target, mesh)
    assert np.allclose(sol[lm], target, atol=1e-9)


def test_identity_mapping_mu_zero(mesh):
    # μ=0（共形恒等）+ 三点固定 → 应还原到单位/自身（近似恒等）
    mu = np.zeros(mesh.face_num, dtype=complex)
    target = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    lm = np.array([np.linalg.norm(mesh.vert - t, axis=1).argmin() for t in target])
    sol = lsqc_solver(mu, lm, target, mesh)
    # 顶点应被保存（只做刚体/仿射拟合，μ=0 时是恒等 + 约束一致）
    assert np.allclose(sol[: len(mesh.vert)], mesh.vert, atol=1e-6)


def test_output_shape_valid(mesh):
    mu = np.zeros(mesh.face_num, dtype=complex)
    target = np.array([[0.0, 0.0], [1.0, 0.0]])
    lm = np.array([np.linalg.norm(mesh.vert - t, axis=1).argmin() for t in target])
    sol = lsqc_solver(mu, lm, target, mesh)
    assert sol.shape == (mesh.vert_num, 2)
    assert np.isfinite(sol).all()
