"""Beltrami 系数计算与截断（μ 域）。"""
import numpy as np

from hbs.mesh import Mesh
from hbs.utils.cast import to_complex


def mu_chop(mu, bound=0.9999, eps=1e-6, constant=None):
    """
    Truncate mu to keep |mu| < 1 while preserving the top-of-range gradient.

    :param mu: Input complex array
    :param bound: Saturation threshold (|mu| < bound 不变)
    :param eps: Safety margin below 1（输出 |mu| ≤ 1-eps）
    :param constant: [兼容] 旧接口，若给出则用旧硬墙行为（constant·mu/|mu|）
    :return: Processed mu
    """
    if constant is not None:
        # 旧接口：硬墙行为（backward compat）
        abs_mu = np.abs(mu)
        idx = abs_mu >= bound
        mu[idx] = constant * (mu[idx] / abs_mu[idx])
        return mu

    abs_mu = np.abs(mu)
    idx = abs_mu >= bound
    if idx.any():
        # 光滑单调饱和：|mu| ∈ [bound, 1-eps]，保序、渐近 1-eps。
        # 修复硬墙把 0.99999/0.99995/0.9999 全压到 0.9999 抹平角部梯度的问题。
        x = abs_mu[idx]
        scale = (1 - eps) - bound
        y = (1 - eps) - scale * np.exp(-(x - bound) / scale)
        mu[idx] = (y / abs_mu[idx]) * mu[idx]  # 保相位
    return mu


def get_beltrami_coefficient(
    mapping: np.ndarray[np.floating], mesh: Mesh
) -> np.ndarray[np.complexfloating]:
    """
    Calculate the Beltrami coefficient of the mapping from "vertex" to "mapping", i.e., f(vertex) = map.
    :param mapping: n x 2 mapped vertex coordinates
    :return: Corresponding Beltrami coefficient
    """
    assert isinstance(mesh, Mesh), "mesh must be Mesh object"

    assert isinstance(mapping, np.ndarray) and np.issubdtype(
        mapping.dtype, np.floating
    ), "mapping must be float array"
    assert mapping.ndim == 2 and mapping.shape == (mesh.vert_num, 2), (
        "mapping must be n x 2 array and n is the number of faces"
    )

    mapping = to_complex(mapping)
    mu = (mesh.Dc * mapping) / (mesh.Dz * mapping)
    mu = mu_chop(mu)
    return mu
