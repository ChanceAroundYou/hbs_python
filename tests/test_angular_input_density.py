"""角速度路径保留原始输入点（不弧长重采样），zipper 仍然重采样。

背景见 `hbs._pipeline.prepare_boundary` 文档：角速度引擎自己按弧长拟合周期样条，
先把输入压到 `circle_point_num` 会提前丢掉几何信息；zipper 需要均匀弧长采样。
"""

from __future__ import annotations

import numpy as np
import pytest

import hbs
from hbs._pipeline import prepare_boundary


def _smooth(n: int = 900, shift: float = 0.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + shift
    r = 1 + 0.25 * np.cos(3 * t) + 0.10 * np.sin(2 * t)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def test_prepare_boundary_can_skip_resampling() -> None:
    """`resample=False` 只做定向与规范锚点：点数不变，点集不变。"""
    b = _smooth(900)
    out = prepare_boundary(b, 500, resample=False)
    assert out.shape == b.shape
    assert np.allclose(np.sort(out.ravel()), np.sort(b.ravel()))


def test_prepare_boundary_default_still_resamples() -> None:
    b = _smooth(900)
    assert prepare_boundary(b, 500).shape == (500, 2)
    assert prepare_boundary(b, 500, resample=True).shape == (500, 2)


def test_skip_resampling_is_still_roll_invariant() -> None:
    """锚点是几何的：不重采样时 roll 后的预处理结果逐位相同。"""
    b = _smooth(900)
    base = prepare_boundary(b, 500, resample=False)
    for k in (1, 37, 211):
        assert np.array_equal(
            prepare_boundary(np.roll(b, k, axis=0), 500, resample=False), base
        )


@pytest.mark.parametrize(("method", "expected"), [("angular", False), ("zipper", True)])
def test_engine_selects_resampling(monkeypatch, method: str, expected: bool) -> None:
    """`get_hbs_result` 按引擎显式选择是否重采样。"""
    import hbs.hbs as H

    seen: list[bool] = []
    real = H.prepare_boundary

    def spy(bound, resample_to, **kwargs):
        seen.append(kwargs.get("resample", True))
        return real(bound, resample_to, **kwargs)

    monkeypatch.setattr(H, "prepare_boundary", spy)
    hbs.get_hbs_result(
        _smooth(600), circle_point_num=500, method=method, normalize="none"
    )
    assert seen == [expected]


def test_angular_result_depends_on_input_density() -> None:
    """稠密输入与它自己的 500 点弧长重采样给出不同的 μ（重采样确有信息损失）。"""
    b = _smooth(900)
    dense = hbs.get_hbs_result(
        b, circle_point_num=500, method="angular", normalize="none"
    )
    coarse = hbs.get_hbs_result(
        prepare_boundary(b, 500),
        circle_point_num=500,
        method="angular",
        normalize="none",
    )
    assert dense.mu.shape == coarse.mu.shape
    assert np.abs(dense.mu - coarse.mu).max() > 1e-8
