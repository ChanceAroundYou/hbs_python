import numpy as np


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


def to_real(x: np.ndarray[np.complexfloating]) -> np.ndarray[np.floating]:
    assert isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.complexfloating), (
        "x must be a complex array"
    )
    assert x.ndim == 1, "x array must be 1D array"
    return np.stack([x.real, x.imag], axis=-1)


def to_complex(x: np.ndarray[np.floating]) -> np.ndarray[np.complexfloating]:
    # assert , "Input must be a numpy array"
    # assert np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer), (
    #     "Input array must have floating point or integer data type"
    # )
    assert isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating), (
        "x must be a real array"
    )
    assert x.ndim == 2 and x.shape[1] == 2, "x must be n x 2 array"
    return x[:, 0] + 1j * x[:, 1]
