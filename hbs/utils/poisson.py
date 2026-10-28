"""稠密 Poisson 核求积：调和延拓的**旧实现**，仅供历史对照实验。

2.1.0 起 `get_hbs` 统一走 `hbs.extension.fem_extend`（P1 稀疏求解）。两者都是
调和延拓，但耗时差 54–63×（密网格 60796 面：473 ms vs 7.6 ms），延拓值差
`1.7e-4 ~ 6.3e-4`，μ 差 `2.0e-2 ~ 4.9e-2`（μ 是导数之比，误差在导数上放大）。

保留它只为让 `experiments/` 里的历史对照脚本可复现；**不要在库代码里使用**。
"""

from __future__ import annotations

import numpy as np

from .cast import to_complex, to_real

def kernel(z, x):
    """
    Compute the Poisson kernel
    :param z: Input points
    :param x: Boundary points
    :return: Poisson kernel
    """
    w = z.reshape(-1, 1) / x.reshape(1, -1)
    w = np.real((1 + w) / (1 - w))
    return w


def integral(
    z: np.ndarray[np.floating],
    x: np.ndarray[np.complexfloating],
    y: np.ndarray[np.complexfloating],
) -> np.ndarray[np.floating]:
    """
    Compute the Poisson integral
    :param z: Internal points to be integrated, n x 2 real array
    :param x: Boundary points, m x 1 complex array
    :param y: Boundary values, m x 1 complex array
    :return: Poisson integral result
    """
    assert z.ndim == 2 and z.shape[1] == 2 and np.issubdtype(z.dtype, np.floating), (
        "z must be n x 2 array with float type"
    )

    assert (
        x.ndim == y.ndim == 1
        and x.shape == y.shape
        and np.issubdtype(x.dtype, np.complexfloating)
        and np.issubdtype(y.dtype, np.complexfloating)
    ), "x and y must be m x 1 array with complex type"

    z = to_complex(z)
    theta = np.angle(x)
    dth = np.diff(theta)
    dth = np.append(dth, theta[0] - theta[-1])
    dth[dth < 0] += 2 * np.pi
    kernel_values = kernel(z, x)
    w = (kernel_values @ (y * dth)) / (2 * np.pi)
    w = np.concatenate([y, w])
    w = to_real(w)
    return w
