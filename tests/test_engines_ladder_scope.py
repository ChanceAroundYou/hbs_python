"""配点阶梯的作用范围：只允许作用在**原始输入**上。

`hbs._engines._rungs` 的几何降级（重采样 / 高斯平滑）本来就是把坏输入改好再试；
如果配点阶梯也跟着一起上，`resample500 + n4096` 会把「换了几何所以过了」伪装成
「加密配点所以过了」——2026-11-10 那版就是这样把两个守卫输入救活的。

这里钉住调用形状：原始级带阶梯，其余降级级一律 `n_ladder=None`（单级严格判据）。
"""
from __future__ import annotations

import numpy as np
import pytest

import hbs._engines as engines
import hbs.harmonic_measure as hm


def _noisy_ellipse(sigma: float = 0.12, seed: int = 0, count: int = 500) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    ellipse = np.stack([1.2 * np.cos(t), np.sin(t)], 1)
    return ellipse + np.random.default_rng(seed).normal(0.0, sigma, (count, 2))


def test_ladder_is_offered_only_to_the_raw_rung(monkeypatch):
    seen: list[tuple[int, object]] = []

    def fake_welding(bound, circle_num=500, n=1024, *, n_ladder=None, **kwargs):
        seen.append((len(bound), n_ladder))
        raise RuntimeError("synthetic failure so every rung is attempted")

    monkeypatch.setattr(hm, "harmonic_measure_welding", fake_welding)
    with pytest.raises(RuntimeError, match="均失败"):
        engines.harmonic_measure(_noisy_ellipse(), 200)

    assert len(seen) == 1 + len(list(engines._rungs(_noisy_ellipse()))) - 1
    ladder_flags = [row[1] is not None for row in seen]
    assert ladder_flags == [True] + [False] * (len(ladder_flags) - 1)
    assert seen[0][1] == engines.HARMONIC_MEASURE_N_LADDER


def test_degradation_rungs_do_not_reach_a_welding_for_the_noisy_input():
    """End-to-end: the noisy guard input must still fail loudly, not be rescued."""
    with pytest.raises(RuntimeError, match="均失败"):
        engines.harmonic_measure(_noisy_ellipse(), 200)
