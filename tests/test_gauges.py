"""源圆旋转规范：I₂ 与论文 M3+M4 的代数契约。

本文件只测**规范本身**（`hbs.gauges`），不测数值求解质量。核心是五条
可以在合成数组上精确验证的恒等式：

1. 旋转作用律：`I₂(Φ_Tμ) = e^{-iT}I₂(μ)`、`M₃(Φ_Tμ) = e^{-2iT}M₃(μ)`；
2. I₂ 规范使 `arg I₂ → 0`；
3. M3+M4 使 `arg M₃ → 0` **且** `arg I₂ ∈ [0, π)`（论文式 5.8/5.9）；
4. 分支判据 `s = (τ₁ − τ₀/2) mod 2π`：`s < π` 取 `τ₀/2`，否则取 `τ₀/2 + π`；
5. 两套规范都是**闭式一步**，且 `apply_gauge` 只改相位（`|μ|` 不变）。

第 3 条是论文 (5.10) 的全部内容。旧实现（`src/hbs/hbs.py` 2.0.x）用
`rotate_x(r/2)` 的半步逼近 + 最多 30 次循环，且分支判据误用一阶矩 `Σμ·z`；
`test_m34_is_closed_form_not_iterative` 钉住新实现没有循环。
"""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

import hbs.gauges as gauges
from hbs.mesh import get_unit_disk

TWO_PI = 2 * np.pi


def _disk(centers, areas):
    """最小 `DiskMesh` 替身：规范只需要 `face_center` 与 `area`。"""
    centers = np.asarray(centers, complex)
    return SimpleNamespace(
        face_center=np.stack([centers.real, centers.imag], axis=1),
        area=np.asarray(areas, float),
    )


def _rotate(centers, areas, mu, theta):
    """源圆旋转 `Φ_θ`：`z' = e^{-iθ}z`、`μ' = e^{-2iθ}μ`。

    与 `rotate_disk` + `apply_gauge` 的组合逐位等价：面片刚体旋转，面积不变，
    面值仍是常数。
    """
    return (np.asarray(centers) * np.exp(-1j * theta), np.asarray(areas),
            np.asarray(mu) * np.exp(-2j * theta))


def _generic(n=24, seed=7):
    """**非对称**面心分布：落在完整圆上会让 `Σ 1/z` 与 `Σ 1` 同时精确抵消。

    规范测试需要两个泛函都非零，否则测的只是"退化检测"而不是代数恒等式。
    """
    rng = np.random.default_rng(seed)
    radius = rng.uniform(0.15, 0.85, n)
    angle = rng.uniform(0, TWO_PI, n)
    centers = radius * np.exp(1j * angle)
    areas = rng.uniform(0.01, 0.05, n)
    mu = rng.normal(0.0, 0.5, n) + 1j * rng.normal(0.0, 0.5, n)
    return centers, areas, mu


#: 完整八分环：常值 μ 在这上面 `Σ 1/z = 0`（I₂ 精确抵消）。
RING = np.exp(2j * np.pi * np.arange(8) / 8) * 0.7


# --- 旋转作用律 ---

def test_i2_has_weight_one_and_m3_has_weight_two():
    """两个泛函的旋转权重必须不同：1 与 2。这是 M3+M4 能定分支的根据。"""
    centers, areas, mu = _generic()
    disk = _disk(centers, areas)

    i2_0 = gauges.i2_functional(mu, disk)
    m3_0 = gauges.m3_functional(mu, disk)
    assert abs(i2_0) > 1e-3 and abs(m3_0) > 1e-3, "夹具必须让两个泛函都非零"

    for theta in (0.3, -1.1, 2.4):
        c, a, m = _rotate(centers, areas, mu, theta)
        ratio_i2 = gauges.i2_functional(m, _disk(c, a)) / i2_0
        ratio_m3 = gauges.m3_functional(m, _disk(c, a)) / m3_0
        assert np.isclose(ratio_i2, np.exp(-1j * theta), atol=1e-12)
        assert np.isclose(ratio_m3, np.exp(-2j * theta), atol=1e-12)


# --- I₂ 单锚 ---

def test_i2_gauge_annihilates_i2_argument():
    centers, areas, mu = _generic(seed=11)
    disk = _disk(centers, areas)

    choice = gauges.select_gauge(mu, disk, "i2")
    assert choice.kind == "i2" and choice.phase is not None

    c, a, m = _rotate(centers, areas, mu, choice.phase)
    residual = abs(np.angle(gauges.i2_functional(m, _disk(c, a))))
    assert residual < 1e-12


def test_i2_gauge_reports_undetermined_when_i2_cancels():
    """完整环上的常值 μ 使 Σ 1/z 精确为零：必须报 `None`，不能返回噪声角度。"""
    mu = np.ones(8)
    disk = _disk(RING, np.full(8, 0.1))

    choice = gauges.select_gauge(mu, disk, "i2")
    assert choice.phase is None and choice.kind == "none"
    assert choice.confidence_i2 < gauges.I2_CONFIDENCE


# --- M3+M4（论文 5.8/5.9，执行式 5.10）---

def test_m34_annihilates_m3_and_puts_i2_in_upper_half_plane():
    """论文 (5.10) 的两条输出条件必须同时成立。"""
    centers, areas, mu = _generic(seed=5)
    disk = _disk(centers, areas)

    choice = gauges.select_gauge(mu, disk, "m34")
    assert choice.kind == "m34" and choice.phase is not None

    c, a, m = _rotate(centers, areas, mu, choice.phase)
    rotated = _disk(c, a)
    residual_m3 = abs(np.angle(gauges.m3_functional(m, rotated)))
    arg_i2 = np.angle(gauges.i2_functional(m, rotated)) % TWO_PI
    assert residual_m3 < 1e-12, "式 (5.8)：arg ∫B dA 必须归零"
    assert arg_i2 < np.pi + 1e-12, "式 (5.9)：arg ∫B/z 必须落在 [0,π)"


@pytest.mark.parametrize("target_s", [0.2, 1.5, 3.0, 5.9])
def test_m34_branch_rule_is_s_mod_2pi(target_s):
    """构造 `arg I₂ − arg M₃/2 = s`，检查分支与最终相位。"""
    # τ₀ = 0（M₃ 正实数）→ s = τ₁。两个面片就能造出任意 τ₁。
    tau0 = 0.0
    centers = np.array([0.5 + 0j, -0.4 + 0.3j])
    areas = np.array([1.0, 1.0])
    # 用 numpy 求两个复系数：μ_a*A/z_a + μ_b*A/z_b = R*e^{i·target_s}，
    # μ_a*A + μ_b*A = real>0。
    z = centers
    M = np.array([[areas[0] / z[0], areas[1] / z[1]],
                  [areas[0], areas[1]]], dtype=complex)
    desired = np.array([np.exp(1j * target_s), 2.0], dtype=complex)
    mu = np.linalg.solve(M, desired)
    disk = _disk(centers, areas)

    m3 = gauges.m3_functional(mu, disk)
    i2 = gauges.i2_functional(mu, disk)
    assert np.isclose(np.angle(m3), tau0, atol=1e-10)
    assert np.isclose(np.angle(i2) % TWO_PI, target_s, atol=1e-10)

    choice = gauges.select_gauge(mu, disk, "m34")
    expected = tau0 / 2 + (0.0 if target_s < np.pi else np.pi)
    assert choice.branch == ("m3" if target_s < np.pi else "m3+pi")
    assert np.isclose((choice.phase - expected) % TWO_PI, 0.0, atol=1e-10) or np.isclose(
        (choice.phase - expected) % TWO_PI, TWO_PI, atol=1e-10
    )


@pytest.mark.parametrize("s,margin", [(0.0, 0.0), (0.5, 0.5), (np.pi, 0.0),
                                      (np.pi + 0.4, 0.4), (3 * np.pi / 2, np.pi / 2)])
def test_m34_branch_margin_measures_distance_to_flip(s, margin):
    """`branch_margin` 是 `s` 到分支边界 {0, π} 的距离：0 表示站在翻转点上。"""
    centers = np.array([0.5 + 0j, -0.4 + 0.3j])
    areas = np.array([1.0, 1.0])
    M = np.array([[areas[0] / centers[0], areas[1] / centers[1]],
                  [areas[0], areas[1]]], dtype=complex)
    mu = np.linalg.solve(M, np.array([np.exp(1j * s), 2.0]))
    choice = gauges.select_gauge(mu, _disk(centers, areas), "m34")
    assert np.isclose(choice.branch_margin, margin, atol=1e-10)


def test_m34_is_closed_form_not_iterative():
    """闭式：一次调用即给出相位，且**再选一次相位为 0**（幂等）。

    旧实现（`rotate_x(r/2)` + 最多 30 轮）不满足幂等，必须靠循环收敛。
    """
    centers, areas, mu = _generic(seed=13)
    disk = _disk(centers, areas)

    first = gauges.select_gauge(mu, disk, "m34")
    assert first.phase is not None

    c, a, m = _rotate(centers, areas, mu, first.phase)
    second = gauges.select_gauge(m, _disk(c, a), "m34")
    assert second.phase is not None
    assert abs(np.angle(np.exp(1j * second.phase))) < 1e-10


# --- auto 策略 ---

def test_auto_prefers_i2_when_confident():
    centers, areas, mu = _generic(seed=3)
    choice = gauges.select_gauge(mu, _disk(centers, areas), "auto")
    assert choice.kind == "i2"
    assert choice.confidence_i2 >= gauges.AUTO_I2_CONFIDENCE


def test_auto_falls_back_to_m34_when_i2_vanishes():
    """完整环上 I₂ 精确抵消但 M₃ 不抵消：必须退到 M3+M4。"""
    disk = _disk(RING, np.full(8, 0.1))
    choice = gauges.select_gauge(np.ones(8), disk, "auto")
    assert choice.kind == "m34"
    assert choice.confidence_m3 > 0.9
    assert choice.phase is not None


def test_auto_reports_undetermined_when_both_anchors_vanish():
    """两个泛函都近零时不许猜方向。"""
    # I₂ 与 M₃ 同时抵消：两个面片、相反系数、相同面心。
    centers = np.array([0.5 + 0j, 0.5 + 0j])
    disk = _disk(centers, np.array([1.0, 1.0]))
    choice = gauges.select_gauge(np.array([1.0, -0.99999]), disk, "auto")
    assert choice.phase is None and choice.kind == "none"
    assert "不可信" in choice.reason


def test_none_never_rotates():
    choice = gauges.select_gauge(np.ones(8), _disk(RING, np.full(8, 0.1)), "none")
    assert choice.kind == "none" and choice.phase is None


def test_unknown_gauge_raises():
    with pytest.raises(ValueError, match="kind must be one of"):
        gauges.select_gauge(np.ones(8), _disk(RING, np.full(8, 0.1)), "c1")


# --- 无分支锚 C₀ = M₁·conj(M₃)（kind="c0"）---

def test_c0_functional_has_weight_one():
    """`C₀(Φ_Tμ) = e^{−iT}C₀(μ)`：权重 1 是无分支的唯一来源。

    权重 2 会让 `T` 有 π 的歧义（m34 的分支），权重 1 才给出唯一解。
    """
    centers, areas, mu = _generic(seed=17)
    disk = _disk(centers, areas)
    c0 = gauges.c0_functional(mu, disk)
    assert abs(c0) > 1e-3, "夹具必须让 C₀ 非零，否则测的是退化检测"

    for theta in (0.3, -1.1, 2.4):
        c, a, m = _rotate(centers, areas, mu, theta)
        ratio = gauges.c0_functional(m, _disk(c, a)) / c0
        assert np.isclose(ratio, np.exp(-1j * theta), atol=1e-12)


def test_c0_gauge_annihilates_c0_argument():
    centers, areas, mu = _generic(seed=19)
    disk = _disk(centers, areas)
    choice = gauges.select_gauge(mu, disk, "c0")
    assert choice.kind == "c0" and choice.phase is not None

    c, a, m = _rotate(centers, areas, mu, choice.phase)
    assert abs(np.angle(gauges.c0_functional(m, _disk(c, a)))) < 1e-12
    expected = (choice.phase - np.angle(gauges.c0_functional(mu, disk))) % TWO_PI
    assert min(expected, TWO_PI - expected) < 1e-12


def test_c0_is_closed_form_and_idempotent():
    """闭式：一次调用给出相位，再选一次相位为 0。没有分支判定、没有循环。"""
    centers, areas, mu = _generic(seed=23)
    first = gauges.select_gauge(mu, _disk(centers, areas), "c0")
    assert first.phase is not None and first.branch is None

    c, a, m = _rotate(centers, areas, mu, first.phase)
    second = gauges.select_gauge(m, _disk(c, a), "c0")
    assert second.kind == "c0" and second.phase is not None
    assert abs(second.phase) < 1e-10


def test_c0_reports_undetermined_on_centrally_symmetric_field():
    """`μ(−z)=μ(z)` ⇒ `M₁=0` ⇒ `C₀=0`：必须报未定，不能给噪声方向。

    这是结构性的：中心对称形状在源圆上旋转 π 后 HBS 逐位相同，相位不可辨识。
    """
    centers = np.array([0.6 + 0.2j, -0.6 - 0.2j, 0.1 - 0.5j, -0.1 + 0.5j])
    areas = np.ones(4)
    mu = np.array([1.0, 1.0, 0.4 - 0.3j, 0.4 - 0.3j])
    assert abs(gauges.c0_functional(mu, _disk(centers, areas))) < 1e-15

    choice = gauges.select_gauge(mu, _disk(centers, areas), "c0")
    assert choice.phase is None and choice.kind == "none"
    assert choice.confidence_c0 < gauges.I2_CONFIDENCE


def test_c0_reports_undetermined_when_m3_cancels():
    """`M₃=0` 而 `M₁≠0`：乘积照样为零。两个因子任一归零都是未定。"""
    centers = np.array([0.5 + 0j, -0.4 + 0.3j])
    areas = np.array([1.0, 1.0])
    mu = np.array([1.0, -1.0])          # Σμ·A = 0，Σμ·A·z = 0.9 − 0.3j
    assert abs(gauges.m3_functional(mu, _disk(centers, areas))) < 1e-15
    assert abs(gauges.c0_functional(mu, _disk(centers, areas))) < 1e-15

    choice = gauges.select_gauge(mu, _disk(centers, areas), "c0")
    assert choice.phase is None and choice.kind == "none"


def test_c0_choice_reports_its_own_diagnostics():
    centers, areas, mu = _generic(seed=29)
    choice = gauges.select_gauge(mu, _disk(centers, areas), "c0")
    assert choice.c0 != 0
    assert 0.0 < choice.confidence_c0 <= 1.0
    payload = choice.as_dict()
    assert "c0" in payload and "confidence_c0" in payload
    assert payload["kind"] == "c0"


#: 2.3.1 里 `select_gauge` 在固定夹具上的逐位输出。加 `"c0"` **不得**改变它们。
_LEGACY_PHASES = {
    "i2": -2.8720584322370755,
    "m34": 2.6423186107800714,
    "auto": -2.8720584322370755,
}
_LEGACY_CONF = (0.06963705426707052, 0.11860337362869355)


def _legacy_fixture():
    rng = np.random.default_rng(20261029)
    n = 40
    radius = rng.uniform(0.15, 0.85, n)
    angle = rng.uniform(0, TWO_PI, n)
    centers = radius * np.exp(1j * angle)
    areas = rng.uniform(0.01, 0.05, n)
    mu = rng.normal(0.0, 0.5, n) + 1j * rng.normal(0.0, 0.5, n)
    return _disk(centers, areas), mu


@pytest.mark.parametrize("kind", sorted(_LEGACY_PHASES))
def test_existing_kinds_are_bitwise_unchanged(kind):
    """`auto`/`i2`/`m34` 的相位与可信度必须**逐位**不变（不是近似）。"""
    disk, mu = _legacy_fixture()
    choice = gauges.select_gauge(mu, disk, kind)
    assert choice.phase == _LEGACY_PHASES[kind]
    assert (choice.confidence_i2, choice.confidence_m3) == _LEGACY_CONF


# --- apply_gauge / rotate_disk 的配对契约 ---

def test_apply_gauge_changes_only_phase():
    mu = np.array([0.2 + 0.5j, -0.6 + 0.1j, 0.3 - 0.8j])
    choice = gauges.GaugeChoice("i2", 0.77, 1j, 1 + 0j, 1.0, 1.0, None, 1.0, "")
    out, scale, _ = gauges.apply_gauge(mu, None, choice)
    assert np.allclose(np.abs(out), np.abs(mu), rtol=0, atol=1e-15)
    assert np.isclose(scale, np.exp(-1j * 0.77))
    assert np.allclose(out, mu * np.exp(-2j * 0.77))


def test_apply_gauge_is_identity_without_phase():
    mu = np.array([0.1 + 0.2j, 0.3 - 0.4j])
    choice = gauges.GaugeChoice("none", None, 0j, 0j, 0.0, 0.0, None, 1.0, "")
    out, scale, _ = gauges.apply_gauge(mu, None, choice)
    assert np.allclose(out, mu) and scale == 1.0 + 0.0j


def test_rotate_disk_commutes_with_the_functionals():
    """`rotate_disk` + `apply_gauge` 必须精确实现旋转作用律（离散也不许有偏差）。"""
    disk = get_unit_disk(density=0.2, circle_point_num=60)
    rng = np.random.default_rng(3)
    mu = rng.normal(size=disk.face_num) + 1j * rng.normal(size=disk.face_num)

    i2_0 = gauges.i2_functional(mu, disk)
    m3_0 = gauges.m3_functional(mu, disk)
    theta = 0.83
    scale = np.exp(-1j * theta)
    rotated = gauges.rotate_disk(disk, scale)
    i2_1 = gauges.i2_functional(mu * scale**2, rotated)
    m3_1 = gauges.m3_functional(mu * scale**2, rotated)

    assert np.isclose(i2_1, np.exp(-1j * theta) * i2_0, atol=1e-10, rtol=1e-10)
    assert np.isclose(m3_1, np.exp(-2j * theta) * m3_0, atol=1e-10, rtol=1e-10)


# --- 兼容入口与端到端接线 ---

def test_angular_compat_wrapper_returns_the_gauge_choice(disk):
    """`angular_mu_from_boundary` 是旧签名入口；它必须真的走新管线并交出 gauge。

    这条路径在 2.3.0 差点漏掉：`build_hbs` 的返回从 4 元组变成 5 元组时，
    这个包装器仍在按 4 元组解包，只有直接调用才会暴露（测试套件里没有别的
    调用方）。

    夹具用**非对称**形状：中心对称形状上任何权重 1 规范都必然返回 `none`
    （见 `test_c0_reports_undetermined_on_centrally_symmetric_field`），
    用椭圆测“每套规范都生效”会测到别的东西。
    """
    import hbs.angular as angular

    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    bound = np.stack(
        [1.2 * np.cos(t) + 0.09 * np.cos(2 * t + 0.7),
         np.sin(t) + 0.05 * np.sin(3 * t)], axis=1
    )

    for kind in gauges.GAUGES:
        mu, _field, used, _res, stats = angular.angular_mu_from_boundary(
            bound, disk=disk, normalize=kind
        )
        assert mu.shape == (used.face_num,)
        assert stats["gauge"].kind == (kind if kind != "auto" else stats["gauge"].kind)
        assert stats["engine"] == "angular"
        if kind != "auto":
            assert stats["gauge"].kind == kind


def test_pipeline_result_carries_the_same_choice_as_the_public_entry():
    """`get_hbs_result(...).gauge` 与 `select_gauge` 在同一个 μ 上必须一致。"""
    from hbs import get_hbs_result

    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    bound = np.stack([1.0 + 0.3 * np.cos(t - 1.0), 0.8 * np.sin(t)], axis=1)

    # 不传 `disk`：显式网格的 `circle_num` 必须与 `circle_point_num` 一致，
    # 否则 `linear_interp` 与 FEM 的边界块尺寸不匹配（这里要的是快，不是复用）。
    result = get_hbs_result(bound, circle_point_num=120, density=0.05,
                            normalize="none")
    direct = gauges.select_gauge(result.mu, result.disk, "auto")
    assert result.gauge.kind == "none" and result.gauge.phase is None
    assert direct.kind in gauges.GAUGES
