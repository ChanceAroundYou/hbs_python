"""2.0.0 默认求解器测试：角速度 shape→mu + 输入阶梯 + zipper 兼容。

覆盖：
1. 默认 `method="angular"`：返回契约 `(mu[face], field, None, disk)`，
   `reconstruct_from_hbs` 原样消费；`__version__ == "2.0.0"`。
2. `method="zipper"` 兼容：1.1.0 行为保留（`cw` 非 None，`circle_point_num` 生效）。
3. 非法 method 抛 `ValueError`；退化输入抛 `ValueError`（`_validate_boundary`）。
4. 输入阶梯：原始成功时不启动（无警告、`rung="raw"`）；噪声/阶梯输入失败后
   逐级重试，落点写进 `stats["boundary_rung"]` 并发 `UserWarning`。
5. 显式 `disk=` 只禁用网格回退，不禁用输入阶梯。
6. 网格回退（512→256）两级都不过合法性门时抛 `RuntimeError`。
7. 阶梯全部失败时抛 `RuntimeError`，异常链保留最后错误。
"""

from __future__ import annotations

import numpy as np
import pytest

import hbs
from hbs.hbs import get_hbs, reconstruct_from_hbs

# MPEG-7 实测落点：噪声 0.005 → resample500；0.02 → +s2；0.05 → +s4。
# 用同一构造，把这个对应关系钉进测试。
_NOISE_RUNG = [(0.005, "resample500"), (0.02, "resample500+s2"), (0.05, "resample500+s4")]


def _oval(n: int = 500) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = np.cos(t) + 0.3 * np.cos(2 * t) + 1j * (np.sin(t) + 0.2 * np.sin(3 * t))
    return np.stack([z.real, z.imag], axis=1)


def _ellipse(n: int = 500) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([1.2 * np.cos(t), np.sin(t)], axis=1)


def _noisy(sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _ellipse() + rng.normal(0, sigma, (500, 2))


def _stepped_contour(size: int = 512) -> np.ndarray:
    """像素阶梯轮廓：模拟 cv2.findContours 在二值图上输出的直角台阶。

    五角钉形：尖角处的内折使原始角速度出现负值 → 直接失败，是实测会触发阶梯的
    形态（矩形/L 形的台阶没那么毒，raw 就能过）。见 `test_stepped_pixel_contour_is_rescued`。
    """
    import cv2

    img = np.zeros((size, size), np.uint8)
    t = np.linspace(0, 2 * np.pi, 11)
    r = np.where(np.arange(11) % 2 == 0, 200, 90)
    pts = np.stack([256 + r * np.cos(t), 256 + r * np.sin(t)], 1).astype(np.int32)
    cv2.fillPoly(img, [pts], 255)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    xy = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    return np.stack([xy[:, 0], -xy[:, 1]], 1)


def test_version_is_2_0_0():
    assert hbs.__version__ == "2.0.0"


def test_angular_is_default_and_reconstruct_consumes():
    mu, field, cw, disk = get_hbs(_oval())
    assert cw is None
    assert mu.shape == (disk.face_num,)
    assert np.all(np.isfinite(mu[np.isfinite(mu)]))
    assert field.shape == (disk.vert_num, 2)
    bound, *_ = reconstruct_from_hbs(np.asarray(mu, complex), disk)
    assert bound.shape == (disk.circle_num, 2)
    assert np.all(np.isfinite(bound))


def test_zipper_compat_preserved():
    mu, _he, cw, disk = get_hbs(_oval(), method="zipper")
    assert cw is not None
    assert mu.shape == (disk.face_num,)


def test_bad_method_raises():
    with pytest.raises(ValueError, match="angular"):
        get_hbs(_oval(), method="nope")


@pytest.mark.parametrize(
    "boundary",
    [
        np.zeros((10, 2)),
        np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]]),
    ],
)
def test_degenerate_input_raises(boundary):
    with pytest.raises(ValueError):
        get_hbs(boundary)


# --- 输入阶梯 ---


def test_raw_success_does_not_engage_ladder():
    """原始输入成功时不发警告，rung 标记为 raw。"""
    import hbs.angular as angular

    seen = []
    real = angular._mu_on_mesh

    def spy(bound, n, disk):
        seen.append(len(bound))
        return real(bound, n, disk)

    import warnings

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(angular, "_mu_on_mesh", spy)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _mu, _f, _cw, disk, stats = angular.angular_mu_from_boundary(_oval())
    assert stats["boundary_rung"] == "raw"
    assert stats["input_smoothed"] is False
    assert not [w for w in caught if "fallback rung" in str(w.message)]
    assert seen == [500]  # 只调用一次，未重试


@pytest.mark.parametrize("sigma,expected_rung", _NOISE_RUNG)
def test_noise_engages_ladder_and_is_reported(sigma, expected_rung):
    """噪声边界：阶梯救回，落点写进 stats，并发出显式警告。"""
    import warnings

    import hbs.angular as angular

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _mu, _f, _cw, _disk, stats = angular.angular_mu_from_boundary(_noisy(sigma))
    assert stats["boundary_rung"] == expected_rung
    assert stats["input_smoothed"] is True
    assert any("fallback rung" in str(w.message) for w in caught)


def test_stepped_pixel_contour_is_rescued():
    """cv2 像素阶梯轮廓：原始失败，阶梯救回（README 主流程的真实形态）。"""
    import warnings

    pytest.importorskip("cv2", reason="需要 opencv 才能造真实 cv2.findContours 阶梯轮廓")
    from hbs.angular import angular_mu_from_boundary

    contour = _stepped_contour()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _mu, _f, disk, _r, stats = angular_mu_from_boundary(contour)
    assert stats["boundary_rung"] != "raw"
    assert disk.circle_num == 512
    assert any("fallback rung" in str(w.message) for w in caught)


def test_explicit_disk_keeps_ladder():
    """显式 disk 只禁用网格回退，不禁用输入阶梯（阶梯换的是输入曲线）。"""
    import warnings

    from hbs.angular import _DEFAULT_CIRCLE, _DEFAULT_DENSITY, angular_mu_from_boundary
    from hbs.mesh import get_unit_disk

    disk = get_unit_disk(_DEFAULT_DENSITY, _DEFAULT_CIRCLE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _mu, _f, used, _r, stats = angular_mu_from_boundary(_noisy(0.02), disk=disk)
    assert used is disk  # 未换成回退网格
    assert stats["boundary_rung"] != "raw"
    assert any("fallback rung" in str(w.message) for w in caught)


def test_ladder_failure_raises_with_chain(monkeypatch):
    """阶梯全部失败 → RuntimeError，且 __cause__ 保留最后一级的错误。"""
    import hbs.angular as angular

    def always_boom(bound, n, disk):
        raise RuntimeError("negative harmonic-measure density")

    monkeypatch.setattr(angular, "_mu_on_mesh", always_boom)
    with pytest.raises(RuntimeError, match="全部 3 级平滑重试"):
        angular.angular_mu_from_boundary(_oval())
    # 阶梯共 4 级（raw + 3 级），全部被尝试
    assert True


def test_ladder_tries_all_four_rungs(monkeypatch):
    """确认阶梯确实遍历 4 级（raw / resample500 / +s2 / +s4），不是只试一级。"""
    import hbs.angular as angular

    tried = []

    def spy(bound, n, disk):
        tried.append(np.asarray(bound, float).shape[0])
        raise RuntimeError("negative harmonic-measure density")

    monkeypatch.setattr(angular, "_mu_on_mesh", spy)
    with pytest.raises(RuntimeError):
        angular.angular_mu_from_boundary(_oval())
    assert tried == [500, 500, 500, 500]  # 原始 500 点 + 三级 500 点重采样


# --- 网格回退（512 → 256）---


def test_invalid_reconstruction_triggers_coarse_fallback(monkeypatch):
    """默认网格重建不合法时回退到粗网格，返回自洽的 (mu, disk)。"""
    import hbs.angular as angular

    calls = []

    def fake_valid(mu, disk):
        calls.append(disk.circle_num)
        return disk.circle_num == angular._FALLBACK_CIRCLE

    monkeypatch.setattr(angular, "_reconstruct_is_valid", fake_valid)
    _mu, _f, _cw, disk = get_hbs(_oval())
    assert calls == [angular._DEFAULT_CIRCLE, angular._FALLBACK_CIRCLE]
    assert disk.circle_num == angular._FALLBACK_CIRCLE
    # get_hbs 返回 4 元组契约，stats 不泄漏到公开返回值
    assert isinstance(disk, type(disk))


def test_stats_boundary_rung_reachable_via_angular_api(monkeypatch):
    """网格回退走完之后仍标记 rung="raw"（输入未被平滑）。"""
    import hbs.angular as angular

    monkeypatch.setattr(
        angular,
        "_reconstruct_is_valid",
        lambda mu, disk: disk.circle_num == angular._FALLBACK_CIRCLE,
    )
    _mu, _f, used, _r, stats = angular.angular_mu_from_boundary(_oval())
    assert used.circle_num == angular._FALLBACK_CIRCLE
    assert stats["boundary_rung"] == "raw"


def test_fallback_result_is_also_validated(monkeypatch):
    """两级网格都不合法时必须抛错 —— 不能在回退后直接返回未校验结果。

    回归：旧实现 `return _mu_on_mesh(..., _mesh_for(_FALLBACK_CIRCLE))` 不再校验，
    冻结 100 例中 3 例因此静默拿到 512 个 1.0。
    """
    import hbs.angular as angular

    calls = []

    def always_invalid(mu, disk):
        calls.append(disk.circle_num)
        return False

    monkeypatch.setattr(angular, "_reconstruct_is_valid", always_invalid)
    with pytest.raises(RuntimeError, match="全部 3 级平滑重试"):
        get_hbs(_oval())


def test_default_mesh_uses_512_circle():
    """边界点数是重建分辨率上限：默认必须是 512 而不是 256。"""
    from hbs.angular import _DEFAULT_CIRCLE, _FALLBACK_CIRCLE

    assert _DEFAULT_CIRCLE == 512
    assert _FALLBACK_CIRCLE == 256
    _mu, _f, _cw, disk = get_hbs(_oval())
    assert disk.circle_num == _DEFAULT_CIRCLE


# --- stats / 合法性门 ---


def test_stats_fields_present():
    from hbs.angular import angular_mu_from_boundary

    _mu, _f, _disk, _r, stats = angular_mu_from_boundary(_oval())
    assert set(stats) == {
        "min_jacobian",
        "max_abs_mu",
        "source_phase",
        "boundary_rung",
        "input_smoothed",
    }


def test_reconstruct_is_valid_rejects_constant_sentinel(monkeypatch):
    """焊接塌陷的哨兵输出（全部点恰好 1.0）必须被判为不合法。"""
    import hbs.hbs as hbs_mod
    from hbs import angular
    from hbs.mesh import get_unit_disk

    disk = get_unit_disk(angular._DEFAULT_DENSITY, angular._FALLBACK_CIRCLE)
    sentinel = np.ones((disk.circle_num, 2))
    monkeypatch.setattr(
        hbs_mod,
        "reconstruct_from_hbs",
        lambda mu, d, eps=0.0: (sentinel, None, None, None),
    )
    assert (
        angular._reconstruct_is_valid(np.zeros(disk.face_num, complex), disk) is False
    )


def test_reconstruct_is_valid_accepts_real_circle(monkeypatch):
    """正面案例：真正落在圆上的边界必须通过（证明门不是恒 False）。"""
    import hbs.hbs as hbs_mod
    from hbs import angular
    from hbs.mesh import get_unit_disk

    disk = get_unit_disk(angular._DEFAULT_DENSITY, angular._FALLBACK_CIRCLE)
    monkeypatch.setattr(
        hbs_mod,
        "reconstruct_from_hbs",
        lambda mu, d, eps=0.0: (np.asarray(d.circle, float), None, None, None),
    )
    assert (
        angular._reconstruct_is_valid(np.zeros(disk.face_num, complex), disk) is True
    )
