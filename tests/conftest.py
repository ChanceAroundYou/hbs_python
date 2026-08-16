"""共享 fixtures：测试用的 mesh/形状。"""
import numpy as np
import pytest

from hbs.mesh import get_unit_disk


@pytest.fixture(scope="session")
def disk():
    """标准单位圆盘（density=0.01, 1000 边界点）——所有 round-trip 共用。"""
    return get_unit_disk(0.01, 1000)


def ellipse(a=1.2, b=1.0, n=500, clockwise=True):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    b = np.stack([a * np.cos(t), b * np.sin(t)], 1)
    return b if not clockwise else b[::-1]


# pytest 之外也便于脚本复用
__all__ = ["disk", "ellipse"]
