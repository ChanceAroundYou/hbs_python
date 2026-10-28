"""μ 的源圆旋转规范：I₂ 与论文 M3+M4（式 5.8/5.9，执行式 5.10）。

两个泛函都是对 `μ` 的面积加权积分：

    M₃(μ) = ∫_D μ(z) dA          ≈ Σ_f μ_f·|A_f|
    I₂(μ) = ∫_D μ(z)/z dA        ≈ Σ_f μ_f·|A_f| / z_f

## 旋转作用（本模块全部符号约定的来源）

源圆旋转 `T`（`ConformalWelding.rotate_x(T)`，等价于 `Φ_T μ = e^{-2iT}μ(e^{iT}·)`）
使两个泛函按**不同权重**变换：

    I₂(Φ_T μ) = e^{-iT}·I₂(μ)        权重 1
    M₃(Φ_T μ) = e^{-2iT}·M₃(μ)       权重 2

权重差来自被积函数：`dA` 在旋转下不变，但 `1/z` 贡献一个 `e^{+iT}`，与 Jacobian 的
`e^{-2iT}` 抵消一半。实测比值 −1.012 / −2.000（`rotate_x` 后重算，1.2% 偏差是重采样
离散误差）。**这正是 M3+M4 必须同时用两个泛函的原因**：M₃ 权重 2 在圆上只能定出
`T mod π`（两个解），需要 I₂ 的权重 1 来挑分支。

## 两套规范

**I₂**（单锚）：`T = arg I₂`。一个方程一个未知量，权重 1 直接给出唯一解，没有
180° 翻转。实现见 `select_gauge(kind="i2")`。

**M3+M4**（论文 Definition 5.8，执行式 5.10）：先由 (5.8) `arg ∫B dA = 0` 定出
`T = τ₀/2 mod π`，再由 (5.9) `arg ∫B/z dA ∈ [0,π)` 挑分支：

    τ₀ = arg M₃ ,  τ₁ = arg I₂ ,  s = (τ₁ − τ₀/2) mod 2π
    T  = τ₀/2           若 s ∈ [0, π)
    T  = τ₀/2 + π       若 s ∈ [π, 2π)

**闭式，一步到位，不需要迭代。** 旧实现 `src/hbs/hbs.py:60-84` 的问题正是这里：
它用 `rotate_x(r/2)` 的**半步**去逼近一个线性方程，把残差放大成 `2r` 而不是消掉，
于是必须靠 30 次循环 + 振荡检测兜底；分支判据还误用了 `arg Σμ·z`（一阶矩 M₁），
论文的 (5.9) 用的是 `∫B/z`，即 I₂。

## 坐标与相位必须**同时**转

规范只选源圆旋转，落到网格上是

    θ = 选定的相位 ,   z' = e^{-iθ}·z ,   μ'(z') = e^{-2iθ}·μ(z)

二次相位来自 `μ = H_z̄/H_z` 的两次导数。因为网格与系数一起刚性旋转，`μ'` 在
旋转后的面心上**逐位等于**原值乘常数——不需要插值。

**只乘常数相位是错的**：那等价于假设 `μ(e^{-iθ}z) = μ(z)`（只有圆才成立），实测
把重建残差从 1e-6 恶化到 1e-3~5e-2。调用方拿到 `scale` 后必须用 `rotate_disk`。

## 为什么 M3+M4 的旋转等变性更好（实测，非本文推导）

`e = μ_T − e^{−2iT}μ₀` 在连续意义下恒为 0。离散网格上它不为零，但**几乎无均值**：
面积加权的角向分解里 m=0 只有 `1e-6`，而 m=1 是 `1e-2`。`M₃ = ∫μ dA` 只看到 m=0，
所以免疫；`I₂ = ∫μ/z dA` 看到的是一阶矩，照单全收。

实测（`runs/焊接规范不变性_2026-10-28/`，解析焊接、N=500）：注入整圈源圆旋转，
总旋转 `T+θ` 的散布 M3+M4 `0.0001–0.0006°`、I₂ `0.90–2.20°`；两套规范都按 O(h²)
随网格加密收敛，比值恒定 48–56×。所以这**不是**"1/z 放大了原点附近面片的离散误差"
——误差在 `r<0.5` 的一阶矩质量反而比 `r∈[0.5,0.9)` 小 8×。也不是"锚越弱方向越抖"
（散播 ∝ `conf^−0.23`，不是 `conf^−1`）。

这个差别**不改变** `auto` 默认选 I₂ 的理由：I₂ 单锚无分支、在正常形状上
（`conf` ≳ 3%）绝对精度已到 1–2°，而 M3+M4 有分支翻转风险（`branch_margin`）。
它只说明 I₂ 的精度余量比 M₃ 小一个量级。

## 退化

两个泛函都可能近零：`|I₂|/Σ|term|` 或 `|M₃|/Σ|term|` 很小时，`arg` 由噪声决定，
旋转只会引入随机相位，且相邻形状之间不连续。这是本项目明确定过的取舍：**近零锚
报告未定，不把数值非零当作稳定**（`研究/C1与I2旋转锚的互补条件.md`）。

`auto` 策略就是为此设计的：I₂ 可信度够高时用 I₂（单锚、无分支、连续）；否则退到
M3+M4，它用 M₃ 的 2 倍权重给出一个**独立**的方向信息，不像 I₂ 那样随形状趋近
旋转对称而整体塌到零。

M3+M4 自身也有病态：`s` 落在 0 或 π 附近时分支选择对扰动敏感（翻分支 = 差 π）。
`GaugeChoice.branch_margin` 报这个裕度，调用方可以据此判定结果是否可信。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hbs.utils.cast import to_complex, to_real

#: 泛函方向的硬可信度门：低于此值连方向都不报，返回 `None`。
I2_CONFIDENCE = 1e-4

#: `auto` 策略的切换门：I₂ 可信度低于此值时改用 M3+M4。
#: 取值依据见 `runs/规范切换_2026-10-28/`：三齿 4e-4、混相 1.8e-2、斜椭圆 4.1e-2、
#: 蛋 4.0e-2、水滴 5.4e-2、双凹 1.0e-1。1e-2 把"纯噪声"与"可用"分开。
AUTO_I2_CONFIDENCE = 1e-2

#: 合法的规范名。
GAUGES = ("auto", "i2", "m34", "none")

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class GaugeChoice:
    """一次规范选择的结果与全部诊断量。

    所有相位都是**已经过分支判定**的 `T`（可直接喂 `rotate_x` 与 `rotate_disk`）。
    """

    kind: str
    """实际施加的规范：`"i2"` / `"m34"` / `"none"`。"""

    phase: float | None
    """源圆旋转角 `T`（弧度）。`None` 表示两个泛函都不可信，未施加旋转。"""

    i2: complex
    """`∫μ/z dA`（原始值，未归一）。"""

    m3: complex
    """`∫μ dA`（原始值，未归一）。"""

    confidence_i2: float
    confidence_m3: float
    """`|泛函| / Σ|term|`：方向的可信度。"""

    branch: str | None
    """M3+M4 的分支：`"m3"`（`T=τ₀/2`）、`"m3+pi"`（`T=τ₀/2+π`），或 `None`。"""

    branch_margin: float
    """`s` 到分支边界 `{0, π}` 的最小距离（弧度），`≥ π/2` 表示分支稳定。

    取 `min(s, |s−π|, 2π−s)`：`s` 落在 0、π、2π 附近时扰动会翻分支。
    """

    reason: str
    """为什么选了这个规范（人类可读）。"""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "phase": None if self.phase is None else float(self.phase),
            "i2": [float(self.i2.real), float(self.i2.imag)],
            "m3": [float(self.m3.real), float(self.m3.imag)],
            "confidence_i2": float(self.confidence_i2),
            "confidence_m3": float(self.confidence_m3),
            "branch": self.branch,
            "branch_margin": float(self.branch_margin),
            "reason": self.reason,
        }


def _wrapped(x: float) -> float:
    """把角度折回 `(−π, π]`。"""
    return float((x + np.pi) % TWO_PI - np.pi)


def _moments(mu: np.ndarray, disk) -> tuple[complex, float, complex, float]:
    """返回 `(I₂, Σ|I₂ term|, M₃, Σ|M₃ term|)`。

    非有限面片与落在原点的面心被跳过（后者除零）。两项共用同一组有效面片，
    这样两个可信度是可比的。
    """
    mu = np.asarray(mu)
    center = to_complex(disk.face_center)
    assert center.shape == (mu.shape[0],)
    area = np.abs(np.asarray(disk.area, dtype=float))
    valid = np.isfinite(mu) & (np.abs(center) > np.finfo(float).eps)
    if not valid.any():
        return 0j, 0.0, 0j, 0.0
    weighted = mu[valid] * area[valid]
    i2_term = weighted / center[valid]
    return (
        complex(i2_term.sum()),
        float(np.abs(i2_term).sum()),
        complex(weighted.sum()),
        float(np.abs(weighted).sum()),
    )


def _confidence(value: complex, magnitude: float) -> float:
    """`|泛函| / Σ|term|`，退化时返回 0 而不是 NaN。"""
    if not np.isfinite(magnitude) or magnitude <= np.finfo(float).eps:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(abs(value) / magnitude)


def _phase_from_arrays(
    mu: np.ndarray,
    face_center: np.ndarray,
    face_area: np.ndarray,
    confidence_threshold: float = I2_CONFIDENCE,
) -> float | None:
    """裸数组版的 `i2_phase`（历史签名，保留给直接吃网格数组的调用方）。

    `select_gauge` 走 `DiskMesh`；这里让只拿到 `(face_center, face_area)` 的
    调用方（与旧实验脚本）不必构造网格。语义与 `i2_phase` 逐位一致。
    """
    mu = np.asarray(mu)
    assert mu.ndim == 1 and face_center.shape == (mu.shape[0], 2)
    assert face_area.shape == mu.shape
    center = to_complex(face_center)
    area = np.abs(np.asarray(face_area, dtype=float))
    valid = np.isfinite(mu) & (np.abs(center) > np.finfo(float).eps)
    if not valid.any():
        return None
    term = mu[valid] * area[valid] / center[valid]
    magnitude = float(np.abs(term).sum())
    if not np.isfinite(magnitude) or magnitude <= np.finfo(float).eps:
        return None
    i2 = complex(term.sum())
    if not np.isfinite(i2) or abs(i2) / magnitude < confidence_threshold:
        return None
    return float(np.angle(i2))


def i2_functional(mu: np.ndarray, disk) -> complex:
    """未加门的 `I₂ = Σ μ·|A|/z`（不可信时调用方需自行判断）。"""
    return _moments(mu, disk)[0]


def m3_functional(mu: np.ndarray, disk) -> complex:
    """未加门的 `M₃ = Σ μ·|A|`（论文式 5.8 的被积式）。"""
    return _moments(mu, disk)[2]


def i2_phase(
    mu: np.ndarray,
    disk,
    confidence_threshold: float = I2_CONFIDENCE,
) -> float | None:
    """`arg I₂(μ)`，当其方向可信时；否则 `None`。

    `None` 的含义是"这个形状没有可靠的归一化方向"，不是错误。
    """
    choice = select_gauge(mu, disk, "i2", confidence_threshold=confidence_threshold)
    return choice.phase


def select_gauge(
    mu: np.ndarray,
    disk,
    kind: str = "auto",
    confidence_threshold: float = I2_CONFIDENCE,
    auto_threshold: float = AUTO_I2_CONFIDENCE,
) -> GaugeChoice:
    """选择源圆旋转 `T` 并返回全部诊断量。**这是唯一入口**。

    :param `kind`: `"auto"` / `"i2"` / `"m34"` / `"none"`
    :param `confidence_threshold`: 硬门，两个泛函都低于此值就返回 `phase=None`
    :param `auto_threshold`: `"auto"` 下改用 M3+M4 的 I₂ 可信度门

    `"m34"` 的闭式解见模块文档。分支判据用 `s = (τ₁ − τ₀/2) mod 2π`，落在
    `[0,π)` 取 `T = τ₀/2`，否则取 `T = τ₀/2 + π`。
    """
    if kind not in GAUGES:
        raise ValueError(f"kind must be one of {GAUGES}, got {kind!r}")
    if kind == "none":
        return GaugeChoice(
            "none", None, 0j, 0j, 0.0, 0.0, None, float(np.pi / 2),
            "normalize='none'：不施加任何规范",
        )

    i2, i2_mag, m3, m3_mag = _moments(mu, disk)
    conf_i2 = _confidence(i2, i2_mag)
    conf_m3 = _confidence(m3, m3_mag)

    tau0 = float(np.angle(m3)) if conf_m3 > 0 else 0.0
    tau1 = float(np.angle(i2)) if conf_i2 > 0 else 0.0
    s = float((tau1 - tau0 / 2.0) % TWO_PI)
    margin = float(min(s, abs(s - np.pi), TWO_PI - s))
    branch = "m3" if s < np.pi else "m3+pi"

    def _i2_choice() -> GaugeChoice:
        return GaugeChoice(
            "i2", float(np.angle(i2)), i2, m3, conf_i2, conf_m3, None, margin,
            f"I₂ 可信度 {conf_i2:.3g} ≥ {auto_threshold:g}",
        )

    def _m34_choice() -> GaugeChoice:
        phase = _wrapped(tau0 / 2.0 + (0.0 if s < np.pi else np.pi))
        why = (
            f"I₂ 可信度 {conf_i2:.3g} < {auto_threshold:g}，退到 M3+M4"
            if kind == "auto"
            else "按请求使用 M3+M4"
        )
        return GaugeChoice(
            "m34", phase, i2, m3, conf_i2, conf_m3, branch, margin,
            f"{why}（分支 {branch}，s={np.degrees(s):.2f}°，裕度 {np.degrees(margin):.2f}°）",
        )

    if kind == "i2":
        if conf_i2 < confidence_threshold:
            return GaugeChoice(
                "none", None, i2, m3, conf_i2, conf_m3, None, margin,
                f"I₂ 可信度 {conf_i2:.3g} < 硬门 {confidence_threshold:g}：未定",
            )
        return _i2_choice()

    if kind == "m34":
        if conf_m3 < confidence_threshold:
            return GaugeChoice(
                "none", None, i2, m3, conf_i2, conf_m3, None, margin,
                f"M₃ 可信度 {conf_m3:.3g} < 硬门 {confidence_threshold:g}：未定",
            )
        return _m34_choice()

    # "auto"
    if conf_i2 >= auto_threshold and conf_i2 >= confidence_threshold:
        return _i2_choice()
    if conf_m3 >= confidence_threshold:
        return _m34_choice()
    return GaugeChoice(
        "none", None, i2, m3, conf_i2, conf_m3, None, margin,
        f"两个泛函都不可信（I₂ {conf_i2:.3g}、M₃ {conf_m3:.3g} < {confidence_threshold:g}）：未定",
    )


def apply_gauge(
    mu: np.ndarray, disk, choice: GaugeChoice
) -> tuple[np.ndarray, complex, GaugeChoice]:
    """按 `choice` 旋转，返回 `(mu_out, scale, choice)`。

    `scale = exp(-i·phase)`，调用方**必须**把它同时用到网格顶点上::

        mu_out, scale, choice = apply_gauge(mu, disk, choice)
        disk_rotated = rotate_disk(disk, scale)

    此后 `(mu_out, disk_rotated)` 与 `(mu, disk)` 描述同一个映射，只在整体旋转下
    不同。`choice.phase is None` 时原样返回（`scale = 1`）。
    """
    if choice.phase is None:
        return np.array(mu, copy=True), 1.0 + 0.0j, choice
    scale = np.exp(-1j * choice.phase)
    return np.asarray(mu) * scale**2, scale, choice


def apply_i2(mu: np.ndarray, disk) -> tuple[np.ndarray, complex, float | None]:
    """I₂ 规范的便捷入口：返回 `(mu_out, scale, phase)`。

    保留为 2.2.0 的公开签名；新代码请用 `select_gauge` + `apply_gauge`，
    它们能同时报出 M3+M4 的诊断量。
    """
    choice = select_gauge(mu, disk, "i2")
    mu_out, scale, _ = apply_gauge(mu, disk, choice)
    return mu_out, scale, choice.phase


def rotate_disk(disk, scale: complex):
    """返回顶点整体乘 `scale` 的新 `DiskMesh`；`face` 数组与顶点顺序不变。

    顶点顺序不变意味着 face 索引一一对应，因此 `apply_gauge` 输出的 μ 可以直接
    配这个网格使用，**不需要插值**。

    原网格上落在 (1,0)/(0,0) 的 landmark 顶点在旋转后已不在原坐标，
    这里把它们的**索引**记到新网格的 `landmark` 属性上，供
    `reconstruct_from_hbs` 使用（旋转是保角变换，钉到顶点自身即等价）。
    """
    from hbs.mesh import DiskMesh

    f = complex(scale)
    circle = to_complex(disk.circle) * f
    in_vert = to_complex(disk.in_vert) * f if disk.in_vert is not None else None
    out_vert = to_complex(disk.out_vert) * f if disk.out_vert is not None else None
    rotated = DiskMesh(
        to_real(circle),
        to_real(in_vert) if in_vert is not None else None,
        to_real(out_vert) if out_vert is not None else None,
        disk.face,
    )
    stored = getattr(disk, "landmark", None)
    if stored is None:
        one_pos = np.all(disk.vert == np.array([1.0, 0.0]), axis=1)
        zero_pos = np.all(disk.vert == np.array([0.0, 0.0]), axis=1)
        stored = np.where(one_pos | zero_pos)[0]
    rotated.landmark = np.asarray(stored, dtype=int)
    return rotated
