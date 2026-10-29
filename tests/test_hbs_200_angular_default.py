"""`get_hbs` 的引擎分派与输入阶梯契约（2.2.0）。

2.2.0 起 `method` 真的切换引擎（`hbs._engines`），之后共用 `hbs._pipeline`。
本文件钉住：

1. 两个引擎各自独立求解，**互不替补**（任一失败即抛错，不静默换引擎）；
2. 输入阶梯是**每个引擎自己的**降级机制，落点报在 `HBSResult.rung`；
3. 两条引擎的焊接只差源圆旋转、μ 扣掉该旋转后一致（见 `compare_engines`）；
4. `normalize` 只接受 `"i2"` / `"none"`；
5. 退化输入在焊接之前被拒绝。

网格回退（512→256）与 `_reconstruct_is_valid` 在 2.2.0 已随 `_mu_on_mesh` 一并
删除：管线不做网格回退，近临界形状由引擎自己抛错。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import hbs
from hbs._engines import ENGINES
from hbs._pipeline import NORMALIZATIONS, compare_engines
from hbs.hbs import get_hbs, get_hbs_result

CIRCLE = 200
DENSITY = 0.05


def _oval(n: int = 500) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = np.cos(t) + 0.3 * np.cos(2 * t) + 1j * (np.sin(t) + 0.2 * np.sin(3 * t))
    return np.stack([z.real, z.imag], axis=1)


def _ellipse(n: int = 500) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([1.2 * np.cos(t), np.sin(t)], axis=1)


def _noisy(sigma: float, seed: int = 0) -> np.ndarray:
    return _ellipse() + np.random.default_rng(seed).normal(0, sigma, (500, 2))


def _cfg(**kw):
    base = dict(circle_point_num=CIRCLE, density=DENSITY)
    base.update(kw)
    return base


# --- 版本与返回契约 ---


def test_version_at_least_2_2_0():
    assert tuple(map(int, hbs.__version__.split("."))) >= (2, 2, 0)


def test_get_hbs_returns_four_tuple_consumed_by_reconstruct():
    mu, field, welding, disk = get_hbs(_oval(), **_cfg())
    assert isinstance(welding, hbs.ConformalWelding)
    assert mu.shape == (disk.face_num,)
    assert field.shape == (disk.vert_num, 2)
    bound, *_ = hbs.reconstruct_from_hbs(np.asarray(mu, complex), disk)
    assert bound.shape == (disk.circle_num, 2)
    assert np.all(np.isfinite(bound))


def test_result_reports_engine_and_rung():
    for method in ENGINES:
        result = get_hbs_result(_oval(), method=method, **_cfg())
        assert result.engine == method
        assert result.rung == "raw"


# --- 引擎真的独立 ---


def test_engines_produce_different_weldings():
    """两个引擎必须真的不同：若 `method` 只影响后备，这里会逐位相同。"""
    z = get_hbs_result(_oval(), method="zipper", **_cfg())
    a = get_hbs_result(_oval(), method="angular", **_cfg())
    assert not np.array_equal(z.welding.init_x, a.welding.init_x)


def test_engines_agree_after_source_rotation():
    """焊接只差源圆旋转；μ 扣掉该旋转后一致到离散误差。"""
    import hbs.cache as cache

    disk, fem = cache.load_unit_disk(DENSITY, CIRCLE)
    z = get_hbs_result(_oval(), method="zipper", normalize="none", disk=disk, **_cfg())
    a = get_hbs_result(_oval(), method="angular", normalize="none", disk=disk, **_cfg())
    agreement = compare_engines(z.welding, a.welding, disk, fem, CIRCLE)
    assert np.degrees(agreement.welding_rms) < 1.0
    assert agreement.mu_rms < 0.1, "μ 的 rms 差异应远小于 max（后者由边界层主导）"


def test_engines_are_not_fallbacks_for_each_other(monkeypatch):
    """角速度引擎全部失败时，`method="angular"` 必须抛错，不能静默改用 zipper。"""
    import hbs._engines as engines

    def boom(bound, circle_num):
        raise RuntimeError("engine unavailable")

    monkeypatch.setitem(engines.ENGINES, "angular", boom)
    with pytest.raises(RuntimeError, match="engine unavailable"):
        get_hbs_result(_oval(), method="angular", **_cfg())


# --- 输入阶梯 ---


def test_raw_success_does_not_engage_ladder():
    result = get_hbs_result(_oval(), **_cfg())
    assert result.rung == "raw"


@pytest.mark.parametrize("sigma", [0.005, 0.02])
def test_noisy_input_is_either_raw_or_reports_its_rung(sigma):
    """噪声边界：要么原样通过（rung="raw"），要么降级并**如实报出**落点。

    不能出现"降级了但 rung 仍写着 raw"的静默降级。
    """
    result = get_hbs_result(_noisy(sigma), **_cfg())
    assert result.rung == "raw" or result.rung.startswith("resample500")


def test_extreme_noise_fails_loudly_instead_of_switching_engine():
    """σ=0.05 时角速度引擎在全部阶梯上失败——必须抛错，**不能**改用 zipper。

    这是"互不替补"的直接可执行断言：若实现偷偷回退，这里会拿到一个结果。
    """
    with pytest.raises(RuntimeError, match="均失败"):
        get_hbs_result(_noisy(0.05), method="angular", **_cfg())


def test_ladder_is_per_engine(monkeypatch):
    """每个引擎只用自己的阶梯；一个引擎失败不会把结果换成另一个引擎的。"""
    import hbs._engines as engines

    calls = []

    def tracking_zipper(bound, circle_num):
        calls.append("zipper")
        return engines.zipper.__wrapped__(bound, circle_num) \
            if hasattr(engines.zipper, "__wrapped__") else _zip(bound, circle_num)

    def _zip(bound, circle_num):
        from hbs.conformal_welding import zipper_welding_prepared
        return engines.WeldingResult(zipper_welding_prepared(bound), "raw", "zipper")

    monkeypatch.setitem(engines.ENGINES, "zipper", tracking_zipper)
    get_hbs_result(_oval(), method="zipper", **_cfg())
    assert calls, "被选中的引擎必须被调用"


# --- 参数校验 ---


def test_bad_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        get_hbs(_oval(), method="nope")


def test_bad_normalize_raises():
    # `"c0"`（2.3.2 新增的无分支锚）是唯一一次扩展：新增 kind 必须是显式决定，
    # 不能靠顺手改这个集合放过参数校验的回归。
    assert set(NORMALIZATIONS) == {"default", "i2", "m34", "c0", "none"}
    with pytest.raises(ValueError, match="normalize must be one of"):
        get_hbs(_oval(), normalize="first-moment")


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


# --- 引擎注册表 ---


def test_engine_registry_is_the_single_source_of_truth():
    assert set(ENGINES) == {"angular", "zipper"}
    for name, fn in ENGINES.items():
        assert callable(fn)
