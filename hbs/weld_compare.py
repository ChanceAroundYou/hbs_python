"""两条共形焊接的**一致性度量**：把焊接视为次数 1 的圆同胚并对齐源圆旋转。

## 为什么需要它

`zipper` 与 `angular` 两条前向路径各自给出一个共形焊接 `f: S¹ → S¹`。
两者应当在**一个源圆旋转**下相同（论文 §5.3/5.4：M1 与 M2 各固定一个 Möbius
转动自由度，剩下的目标圆旋转由 I₂ 规范吸收）。要判断"两个引擎是否一致"，
必须先扣掉这个旋转，否则会把纯规范自由度误报成映射差异。

## 焊接的函数形式

`ConformalWelding` 存的是两条边界上的点 `(x_i, y_i)`，按边界点配对。写成角度
并展开成连续分支后得到次数 1 的圆同胚

    f(θ) = θ + α(θ),    f(θ + 2π) = f(θ) + 2π,

其中 `α` 有界。归一化取 `f(0) = 0`（与论文 §6.3 的 `f(1)=1` 同义，只是在
圆上换了起点）。

## 源圆旋转的正确作用

旋转源圆 `x → e^{iT}x` 不改变 `(x_i, y_i)` 这个**点集**，只改变参数标签。
在此之上重新做 `f(0)=0` 归一化，得到

    f_T(θ) = f(θ + T) - f(T).

注意**不是** `f(θ - T)`：那会漏掉 `-f(T)` 这个常数项，而该常数项正是
目标圆规范的一部分。这一条是本模块的主要易错点，`tests/test_weld_compare.py`
里有构造性自校验钉住它（恢复 `T₀` 到 1e-3 弧度以内）。

## 度量

对两条焊接 `f₁, f₂`：

    T*  = argmin_T  rms( wrap( f₁(θ+T) - f₁(T) - f₂(θ) ) )
    δ   = wrap( f₁(θ+T*) - f₁(T*) - f₂(θ) )        ∈ (-π, π]
    S   = mean δ                                    目标圆规范残差
    本质差距 = rms(δ - S)

`S` 是两条焊接在目标圆上相差的转动量。它**不应**算作两引擎的分歧：`I₂` 规范
（`hbs.gauges`）做的正是把目标圆的旋转钉住，而 `μ = H_z̄/H_z` 对边界值的常数
相位因子不敏感。所以真正的判据是扣掉 `S` 之后的 `rms(δ-S)`。

`mean|δ|` 一并返回：它是"不做任何规范修正时两条焊接的实际平均差"，用于对照。

## 测量口径

`f` 在陡峭段（形状的尖角对应焊接的近似跳跃）斜率可达 10 以上，此时
`max|Δf|` 会把 `θ` 上 `1e-3` 弧度的误差放大成数十度。因此：

- 优化目标用 `rms`（或 `mean`），**不要**用 `max`；
- 对齐必须在连续的 `T` 上搜索，不能只在样本格点上取 `np.roll`；
- 报告 `max|δ-S|` 只作诊断，不参与判定。

实测口径见 `runs/焊接对齐度量_2026-10-28/`：N=1200 时两条焊接的
`rms(δ-S) ≈ 0.03°`，且随 N 按 O(1/N) 收敛。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 判定"这是一个次数 1 的圆同胚"的容差（弧度）。超出即说明焊接退化。
DEGREE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class WeldingCurve:
    """次数 1 的圆同胚 `f(θ) = θ + α(θ)`，归一化 `f(0) = 0`。

    `source` 是源圆角（单调、自 0 起、长度 n）；`target` 是连续展开的目标角
    （长度 n+1，末项与首项相差恰好 2π，对应闭合回原点的那个采样）。

    构造时即校验，避免把退化的焊接带进度量（否则只会得到无意义的统计量）。
    注意 `target` 是**分段线性**的：两个不同采样网格之间的比较会有 O(h²) 的
    插值底噪，`compare_welding` 的残差不会低于这个量级。
    """

    source: np.ndarray
    target: np.ndarray

    def __post_init__(self):
        source = np.asarray(self.source, float)
        target = np.asarray(self.target, float)
        if source.ndim != 1 or target.ndim != 1:
            raise ValueError("source and target must be 1D")
        if source.size < 4:
            raise ValueError("welding needs at least 4 samples")
        if target.size != source.size + 1:
            raise ValueError(
                f"target must have len(source)+1 = {source.size + 1} closed samples, "
                f"got {target.size}"
            )
        if not (np.all(np.isfinite(source)) and np.all(np.isfinite(target))):
            raise ValueError("welding contains non-finite values")
        if np.unique(np.round(source, 12)).size != source.size:
            raise ValueError("source angles are not distinct; welding is degenerate")
        if np.any(np.diff(source) <= 0):
            raise ValueError("source angles must be strictly increasing")
        span = float(target[-1] - target[0])
        if abs(span - 2 * np.pi) > DEGREE_TOLERANCE:
            raise ValueError(
                f"not a degree-1 circle homeomorphism: target advance {span:.9f} != 2π"
            )
        if np.any(np.diff(target) <= 0):
            raise ValueError("target angles must be strictly increasing (orientation +1)")
        if abs(target[0]) > DEGREE_TOLERANCE:
            raise ValueError(f"welding must be normalized to f(0)=0, got {target[0]:.3e}")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)

    def __call__(self, theta) -> np.ndarray:
        """在任意角上求值，满足 `f(θ + 2πn) = f(θ) + 2πn`。"""
        theta = np.asarray(theta, float)
        turns = np.floor(theta / (2 * np.pi))
        knots = np.r_[self.source, 2 * np.pi]
        return np.interp(np.mod(theta, 2 * np.pi), knots, self.target) + 2 * np.pi * turns


def analytic_welding(alpha, samples: int = 2001) -> WeldingCurve:
    """由解析相位 `f(θ) = θ + alpha(θ)` 构造焊接，`alpha` 须 2π-周期且保持单调。

    用于把对齐数学从插值底噪里分离出来：解析构造的 `f` 与采样密度无关，
    因此自校验可以要求机器精度。
    """
    theta = np.linspace(0, 2 * np.pi, int(samples), endpoint=False)
    values = theta + np.asarray(alpha(theta), float)
    if abs(values[0]) > 1e-12:
        raise ValueError(f"alpha must vanish at 0 for a normalized welding, got {values[0]}")
    wrap = float(np.asarray(alpha(np.array([2 * np.pi])))[0])
    if abs(wrap - values[0]) > 1e-9:
        raise ValueError(f"alpha must be 2π-periodic; alpha(2π)-alpha(0) = {wrap:.3e}")
    if np.any(np.diff(values) <= 0):
        raise ValueError("alpha does not preserve monotonicity (need 1 + alpha' > 0)")
    return WeldingCurve(source=theta, target=np.r_[values, 2 * np.pi])


@dataclass(frozen=True)
class WeldingAgreement:
    """两条焊接的对齐结果（角度单位：弧度）。"""

    shift: float
    """对齐所需的**源圆**旋转 `T*`。"""

    gauge: float
    """对齐后目标圆上的平均残差 `S`；这是 `I₂` 规范应吸收的量。"""

    mean_abs: float
    """`mean|δ|`：不做目标圆规范修正时的平均辐角差。"""

    rms_residual: float
    """`rms(δ - S)`：扣掉目标圆规范后的**本质差距**，判定用这个。"""

    max_residual: float
    """`max|δ - S|`：仅作诊断（陡峭段会放大，不参与判定）。"""

    residual: np.ndarray
    grid: np.ndarray


def welding_curve(cw) -> WeldingCurve:
    """从 `ConformalWelding` 构造 `WeldingCurve`。

    要求 `x`、`y` 都在单位圆上且按同一个物理边界点配对（`zipper_welding_prepared`
    与 `ConformalWelding.from_angular` 都满足）。源角按升序排列，目标角用
    `unwrap` 展开；若整体绕向相反则翻转，保证次数为 `+1`。
    """
    a = np.mod(np.asarray(cw.init_x_angle, float), 2 * np.pi)
    b = np.mod(np.angle(np.asarray(cw.init_y, complex)), 2 * np.pi)
    if a.shape != b.shape or a.ndim != 1 or a.size < 4:
        raise ValueError("welding must be two equal-length 1D angle arrays (n >= 4)")
    order = np.argsort(a)
    a, b = a[order], b[order]
    closed = np.unwrap(np.r_[b, b[0]])
    if closed[-1] < closed[0]:
        closed = -closed
    closed -= closed[0]
    # 采样网格自己的绕向/起点校正后，把跨度精确归到 2π：`np.unwrap` 的末项
    # 与 2π 只差浮点舍入（~1e-16），不归一化会踩到 WeldingCurve 的容差门。
    closed *= 2 * np.pi / closed[-1]
    return WeldingCurve(source=a - a[0], target=np.asarray(closed, float))


def compare_welding(
    first: WeldingCurve,
    second: WeldingCurve,
    grid: int = 4096,
    coarse: int = 2880,
    refine: int = 401,
) -> WeldingAgreement:
    """对齐两条焊接并返回一致性统计。

    :param `first`: 被比较的焊接（对齐目标是让 `second` 去贴合它）
    :param `second`: 参考焊接
    :param `grid`: 求值网格点数（`rms` 的积分节点）
    :param `coarse`: 源圆旋转的粗搜索点数（默认 2880 → 0.125°）
    :param `refine`: 粗搜索最优点附近的精修点数
    """
    theta = np.linspace(0, 2 * np.pi, int(grid), endpoint=False)

    def residual(shift: float) -> np.ndarray:
        shifted = first(theta + shift) - first(np.array([shift]))[0]
        return np.mod(shifted - second(theta) + np.pi, 2 * np.pi) - np.pi

    def score(shift: float) -> float:
        return float(np.sqrt(np.mean(residual(shift) ** 2)))

    candidates = np.linspace(0, 2 * np.pi, int(coarse), endpoint=False)
    scores = np.array([score(t) for t in candidates])
    best = float(candidates[int(np.argmin(scores))])
    # 两级精修：每次把窗口缩到上一步格距，使 `T*` 的分辨率从 `2π/coarse` 降到
    # `2π/(coarse·refine²)`。只做一级时自比对会卡在 ~1.6e-6 的假残差上（那是
    # `T*` 的量化误差经 `rms` 反映出来的下界，不是焊接差异）。
    window = 2 * np.pi / int(coarse)
    for _ in range(2):
        fine = np.linspace(best - window, best + window, int(refine) | 1)
        values = np.array([score(t) for t in fine])
        best = float(fine[int(np.argmin(values))])
        window *= 2.0 / (int(refine) | 1)
    shift = best

    delta = residual(shift)
    gauge = float(delta.mean())
    deviation = delta - gauge
    return WeldingAgreement(
        shift=shift,
        gauge=gauge,
        mean_abs=float(np.abs(delta).mean()),
        rms_residual=float(np.sqrt(np.mean(deviation**2))),
        max_residual=float(np.abs(deviation).max()),
        residual=deviation,
        grid=theta,
    )


def compare_conformal_welding(first, second, **kwargs) -> WeldingAgreement:
    """`compare_welding` 的便捷入口：直接吃两个 `ConformalWelding` 对象。"""
    return compare_welding(welding_curve(first), welding_curve(second), **kwargs)
