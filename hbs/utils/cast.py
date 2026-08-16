"""数组格式转换工具（real ↔ complex）。"""
import numpy as np


def to_real(x: np.ndarray[np.complexfloating]) -> np.ndarray[np.floating]:
    assert isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.complexfloating), (
        "x must be a complex array"
    )
    assert x.ndim == 1, "x array must be 1D array"
    return np.stack([x.real, x.imag], axis=-1)


def to_complex(x: np.ndarray[np.floating]) -> np.ndarray[np.complexfloating]:
    assert isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating), (
        "x must be a real array"
    )
    assert x.ndim == 2 and x.shape[1] == 2, "x must be n x 2 array"
    return x[:, 0] + 1j * x[:, 1]
