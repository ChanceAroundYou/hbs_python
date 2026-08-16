"""Möbius 变换单元测试（单位圆盘自同构性质）。"""
import numpy as np

from hbs.utils.mobius import mobius, mobius_d, mobius_inv


def _unit_circle(n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.exp(1j * t)


def test_identity_at_a_zero():
    z = np.array([0.1 + 0.2j, -0.5 + 0.7j, 0.9 - 0.3j])
    assert np.allclose(mobius(z, 0), z)


def test_theta_rotation_at_a_zero():
    z = np.array([0.5 + 0.5j, -0.3 + 0.1j])
    out = mobius(z, 0, theta=0.7)
    assert np.allclose(out, z * np.exp(0.7j))


def test_unit_disk_stays_unit_disk():
    # |a|<1 的自同构：圆内→圆内，圆周→圆周
    a = 0.3 + 0.2j
    interior = np.array([0.0, 0.2 + 0.1j, -0.4 - 0.3j])
    assert np.abs(mobius(interior, a)).max() < 1.0
    assert np.allclose(np.abs(mobius(_unit_circle(), a)), 1.0, atol=1e-9)


def test_inverse_composes_to_identity():
    for a in (0.0, 0.2j, -0.3 + 0.4j, 0.5 + 0.1j):
        z = np.array([0.1 + 0.1j, -0.4 + 0.6j, 0.8j])
        assert np.allclose(mobius_inv(mobius(z, a), a), z, atol=1e-9)


def test_inverse_with_theta():
    a = 0.2 + 0.3j
    theta = 0.5
    z = np.array([0.3 + 0.2j, -0.1 + 0.4j])
    w = mobius(z, a, theta)
    assert np.allclose(mobius_inv(w, a, theta), z, atol=1e-9)


def test_derivative_matches_finite_difference():
    a = 0.2 + 0.1j
    z0 = 0.3 + 0.2j
    theta = 0.4
    h = 1e-6
    # mobius 只接受数组输入（内部有 inf/极点 item-assignment）
    num = (mobius(np.array([z0 + h]), a, theta)[0] - mobius(np.array([z0 - h]), a, theta)[0]) / (2 * h)
    assert np.isclose(mobius_d(np.array([z0]), a, theta)[0], num, rtol=1e-5)


def test_pole_handling():
    a = 0.5
    pole = 1 / np.conj(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = mobius(np.array([pole]), a)
        at_inf = mobius(np.array([np.inf]), a)
    assert np.isinf(out[0])
    # 无穷远处 → -k / conj(a)
    assert np.allclose(at_inf, -1 / np.conj(a))
