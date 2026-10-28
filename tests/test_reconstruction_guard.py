"""重建合法性门与一次性粗网格回退（`validate_reconstruction`）。

来源：研究仓 1.2.0 vendored 角速度路径的 `_reconstruct_is_valid` 与
`(0.04, 512) → (0.04, 256)` 回退阶梯（2026-10-24 修复「回退结果未复检」后定稿）。
2.3.2 移入库，默认关闭，语义保持不变：

- 门只认「有限、形状正确、展布 > DEGENERATE_EXTENT」的边界；
- 默认网格不合法才回退一次，回退结果同样过门；
- 两级都不合法抛 `RuntimeError`，不返回未校验结果；
- 显式传 `disk` 时不做门与回退（与迁移前一致）。
"""

from __future__ import annotations

import numpy as np
import pytest

import hbs
import hbs._guard as guard
import hbs.hbs as hbs_mod
from hbs.mesh import get_unit_disk

CIRCLE = 120
DENSITY = 0.05


def _oval(n: int = 250) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = np.cos(t) + 0.3 * np.cos(2 * t) + 1j * (np.sin(t) + 0.2 * np.sin(3 * t))
    return np.stack([z.real, z.imag], axis=1)


def _disk() -> "hbs.DiskMesh":
    return get_unit_disk(DENSITY, CIRCLE)


def test_guard_is_off_by_default(monkeypatch):
    """默认不开门：不得多做一次重建。"""
    from hbs._guard import reconstruction_is_valid

    calls = []

    def boom(mu, disk):
        calls.append(disk.circle_num)
        raise AssertionError("默认不应调用合法性门")

    monkeypatch.setattr(guard, "reconstruction_is_valid", boom)
    result = hbs.get_hbs_result(
        _oval(), circle_point_num=CIRCLE, density=DENSITY
    )
    assert calls == []
    assert result.guard_rung is None
    assert reconstruction_is_valid is not boom


def test_guard_keeps_default_mesh_when_valid(monkeypatch):
    calls = []

    monkeypatch.setattr(
        guard, "reconstruction_is_valid",
        lambda mu, disk: calls.append(disk.circle_num) or True,
    )
    result = hbs.get_hbs_result(
        _oval(), circle_point_num=CIRCLE, density=DENSITY,
        validate_reconstruction=True,
    )
    assert calls == [CIRCLE]
    assert result.guard_rung is None
    assert result.disk.circle_num == CIRCLE


def test_guard_falls_back_to_half_circle(monkeypatch):
    calls = []

    monkeypatch.setattr(
        guard, "reconstruction_is_valid",
        lambda mu, disk: calls.append(disk.circle_num) or disk.circle_num == 64,
    )
    result = hbs.get_hbs_result(
        _oval(), circle_point_num=CIRCLE, density=DENSITY,
        validate_reconstruction=True,
    )
    assert calls == [CIRCLE, 64]
    assert result.guard_rung == "fallback:64"
    assert result.disk.circle_num == 64
    assert result.mu.shape == (result.disk.face_num,)


def test_guard_raises_when_both_meshes_degenerate(monkeypatch):
    calls = []

    monkeypatch.setattr(
        guard, "reconstruction_is_valid",
        lambda mu, disk: calls.append(disk.circle_num) or False,
    )
    with pytest.raises(RuntimeError, match="退化为一点"):
        hbs.get_hbs_result(
            _oval(), circle_point_num=CIRCLE, density=DENSITY,
            validate_reconstruction=True,
        )
    assert calls == [CIRCLE, 64]


def test_guard_skips_explicit_disk(monkeypatch):
    """显式网格由调用方负责：不门、不回退。"""
    monkeypatch.setattr(
        guard, "reconstruction_is_valid",
        lambda mu, disk: (_ for _ in ()).throw(AssertionError("不应调用")),
    )
    result = hbs.get_hbs_result(
        _oval(), circle_point_num=CIRCLE, disk=_disk(), validate_reconstruction=True
    )
    assert result.guard_rung is None
    assert result.disk.circle_num == CIRCLE


def test_guard_does_not_change_a_valid_result():
    """门通过时，结果与不开门逐位相同（同一网格、同一求解）。"""
    plain = hbs.get_hbs_result(_oval(), circle_point_num=CIRCLE, density=DENSITY)
    guarded = hbs.get_hbs_result(
        _oval(), circle_point_num=CIRCLE, density=DENSITY,
        validate_reconstruction=True,
    )
    assert np.array_equal(plain.mu, guarded.mu)
    assert np.array_equal(plain.disk.circle, guarded.disk.circle)


def test_guard_refuses_below_fallback_floor(monkeypatch):
    """源圆已到下限时不做无意义的重试，直接报错。"""
    monkeypatch.setattr(guard, "reconstruction_is_valid", lambda mu, disk: False)
    with pytest.raises(RuntimeError, match="回退下限"):
        hbs.get_hbs_result(
            _oval(), circle_point_num=60, density=0.1,
            validate_reconstruction=True,
        )


# --- 门本身 ---------------------------------------------------------------


def _patch_reconstruct(monkeypatch, value):
    monkeypatch.setattr(
        hbs_mod, "reconstruct_from_hbs", lambda mu, d, eps=0.0: (value, None, None, None)
    )


def test_gate_rejects_constant_sentinel(monkeypatch):
    disk = _disk()
    _patch_reconstruct(monkeypatch, np.ones((disk.circle_num, 2)))
    assert guard.reconstruction_is_valid(np.zeros(disk.face_num, complex), disk) is False


def test_gate_rejects_nonfinite(monkeypatch):
    disk = _disk()
    bad = np.ones((disk.circle_num, 2))
    bad[3, 1] = np.nan
    _patch_reconstruct(monkeypatch, bad)
    assert guard.reconstruction_is_valid(np.zeros(disk.face_num, complex), disk) is False


def test_gate_rejects_wrong_shape(monkeypatch):
    disk = _disk()
    _patch_reconstruct(monkeypatch, np.zeros((disk.circle_num - 1, 2)))
    assert guard.reconstruction_is_valid(np.zeros(disk.face_num, complex), disk) is False


def test_gate_accepts_a_real_circle(monkeypatch):
    disk = _disk()
    _patch_reconstruct(monkeypatch, np.asarray(disk.circle, float))
    assert guard.reconstruction_is_valid(np.zeros(disk.face_num, complex), disk) is True


def test_gate_rejects_when_reconstruction_raises(monkeypatch):
    disk = _disk()

    def boom(mu, d, eps=0.0):
        raise ValueError("near-critical mu")

    monkeypatch.setattr(hbs_mod, "reconstruct_from_hbs", boom)
    assert guard.reconstruction_is_valid(np.zeros(disk.face_num, complex), disk) is False


def test_gate_agrees_with_the_real_pipeline_on_a_healthy_shape():
    """不打桩：正常形状的真实重建必须过门。

    反面的「常数哨兵」由 `test_gate_rejects_constant_sentinel` 打桩覆盖——
    注意恒零 μ 的重建是合法圆，不能用它当退化样例。
    """
    result = hbs.get_hbs_result(_oval(), circle_point_num=CIRCLE, density=DENSITY)
    assert guard.reconstruction_is_valid(result.mu, result.disk) is True
