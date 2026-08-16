"""Poisson 积分（调和延拓）测试。"""
import numpy as np

from hbs.mesh import Mesh
from hbs.utils.poisson import integral


def _grid(density=0.25):
    """小矩形网格，竖/横坐标准备好一个内部 z。"""
    import numpy as np
    from hbs.mesh import get_rect

    return get_rect(width=1, height=1, density=density)


def test_constant_boundary_gives_constant_interior():
    # 边界值全为常数 c → Poisson 积分内部也应为 c（调和函数的最大值原理/常值）
    m = _grid()
    c = 1j * 2.0
    x = np.exp(np.linspace(0, 2 * np.pi, 64, endpoint=False) * 1j)  # 单位圆边界
    y = np.full(64, c)
    # 内部点 z=0 处（远离边界），数值上接近 c
    z_in = np.array([[0.0, 0.0]])
    h = integral(z_in, x, y)
    assert np.allclose(h[-1], [c.real, c.imag], atol=0.35)


def test_returns_boundary_appended():
    # integral 输出 = 边界值 + 内部值（n+m 行）
    x = np.exp(np.linspace(0, 2 * np.pi, 32, endpoint=False) * 1j)
    y = np.exp(np.arange(32) * 0.1j)
    z_in = np.array([[0.1, -0.2], [0.3, 0.4]])
    h = integral(z_in, x, y)
    assert h.shape == (32 + 2, 2)
    assert np.allclose(h[:32], np.stack([y.real, y.imag], 1))
