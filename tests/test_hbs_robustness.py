"""Regression tests for HBS numerical-boundary robustness."""

import numpy as np
import pytest

from hbs import get_hbs, reconstruct_from_hbs
from hbs.gauges import _phase_from_arrays as lambda_phase


def _ellipse(n=500):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(t), 0.5 * np.sin(t)], axis=1)


def test_noisy_boundary_recovers_with_finite_roundtrip(disk):
    """Failure-only smoothing rescues the known zipper failure at sigma=0.02."""
    noisy = _ellipse() + np.random.default_rng(0).normal(0, 0.02, (500, 2))

    hbs, _, _, result_disk = get_hbs(noisy, 1000, 0.01, disk)
    boundary, in_points, out_points, _ = reconstruct_from_hbs(hbs, result_disk)

    assert np.isfinite(hbs).all()
    assert np.isfinite(boundary).all()
    assert np.isfinite(in_points).all()
    assert np.isfinite(out_points).all()


@pytest.mark.parametrize(
    "boundary",
    [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.zeros((3, 2)),
        np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]]),
    ],
)
def test_degenerate_boundary_raises_clear_value_error(boundary):
    with pytest.raises(ValueError, match="boundary"):
        get_hbs(boundary, 100, 0.05)


def test_lambda_phase_uses_area_weighted_confidence_gate():
    """A nearly cancelling I₂ has no trustworthy normalization direction."""
    hbs = np.array([1.0 + 0j, -0.99995 + 0j])
    centers = np.array([[1.0, 0.0], [1.0, 0.0]])
    areas = np.array([1.0, 1.0])

    assert lambda_phase(hbs, centers, areas, 1e-4) is None


def test_lambda_phase_returns_angle_for_confident_weighted_moment():
    hbs = np.array([1.0j, 1.0j])
    centers = np.array([[1.0, 0.0], [1.0, 0.0]])
    areas = np.array([1.0, 2.0])

    assert np.isclose(lambda_phase(hbs, centers, areas, 1e-4), np.pi / 2)


def test_zipper_gauge_is_one_shot_not_iterated():
    """2.1.0 起 zipper 不再有 8 轮 rotate_x 循环；规范化是一次刚体变换。

    旧循环用 `rotate_x(r)` + `linear_interp` 逼近 arg I₂=0，重采样到固定 x
    网格导致结果依赖输入起点（roll 残差 0.46–0.87）。这里钉住"不再有循环"
    这一契约：模块里不应再出现迭代上限常量。
    """
    import hbs.hbs as mod

    assert not hasattr(mod, "_LAMBDA_MAX_ITERATIONS")
