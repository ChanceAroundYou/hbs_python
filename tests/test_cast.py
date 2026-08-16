"""to_real / to_complex 数组转换单元测试。"""
import numpy as np
import pytest

from hbs.utils.cast import to_complex, to_real


def test_to_real_values_and_shape():
    z = np.array([1 + 2j, -3 + 0.5j, -1j])
    out = to_real(z)
    assert out.shape == (3, 2)
    assert np.allclose(out[:, 0], [1, -3, 0])
    assert np.allclose(out[:, 1], [2, 0.5, -1])


def test_to_complex_values_and_shape():
    x = np.array([[1.0, 2.0], [-3.0, 0.5], [0.0, -1.0]])
    out = to_complex(x)
    assert out.shape == (3,)
    assert np.allclose(out, [1 + 2j, -3 + 0.5j, -1j])


def test_roundtrip():
    z = np.array([0.1 + 0.2j, -1.5 + 3j, 2j])
    assert np.allclose(to_complex(to_real(z)), z)


def test_to_real_rejects_float_input():
    with pytest.raises(AssertionError):
        to_real(np.array([1.0, 2.0]))


def test_to_real_rejects_2d():
    with pytest.raises(AssertionError):
        to_real(np.array([[1 + 1j, 2 + 2j]]))


def test_to_complex_rejects_complex_input():
    with pytest.raises(AssertionError):
        to_complex(np.array([1 + 1j, 2 + 2j]))


def test_to_complex_rejects_wrong_width():
    with pytest.raises(AssertionError):
        to_complex(np.zeros((3, 3)))
