"""zipper / ConformalWelding 基础测试。"""
import numpy as np

from hbs.conformal_welding import get_conformal_welding
from hbs.utils.geodesic_welding import geodesic_welding
from hbs.utils.zipper import zipper


def _ellipse_complex(n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return 1.2 * np.cos(t) + 1j * np.sin(t)


def test_zipper_outputs_on_unit_disk():
    b = _ellipse_complex()
    x, _, _ = zipper(b)
    assert x.shape[0] == len(b)
    assert np.all(np.isfinite(x))
    assert np.all(np.abs(x) <= 1 + 1e-9)


def test_get_conformal_welding_stable():
    # get_conformal_welding 需要 (n,2) 实边界（内部转复数）
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    b = np.stack([1.2 * np.cos(t), np.sin(t)], 1)[::-1]
    cw = get_conformal_welding(b)
    assert cw.x.shape[0] == 200
    assert np.all(np.abs(cw.x) <= 1 + 1e-9)


def test_geodesic_welding_shape_and_return():
    nn = 50
    s = np.exp(1j * np.linspace(0, 2 * np.pi, nn, endpoint=False))
    t = np.exp(1j * np.linspace(0, 2 * np.pi, nn, endpoint=False) * 1.3)
    rng = np.random.default_rng(0)
    a = rng.normal(size=nn) + 1j * rng.normal(size=nn)
    b = rng.normal(size=nn) + 1j * rng.normal(size=nn)
    out_a, out_b = geodesic_welding(a, b, s, t)
    assert out_a.shape[0] == len(b)
    assert out_b.shape[0] == len(b)
    assert np.all(np.isfinite(out_a)) and np.all(np.isfinite(out_b))
