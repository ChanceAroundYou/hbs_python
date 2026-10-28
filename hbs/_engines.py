"""两套**独立**的共形焊接引擎。

## 为什么必须是两套

`get_hbs(method=...)` 的 `method` 必须真的切换算法，否则它是个骗人的参数，
而且失去交叉验证能力：若角速度解在某类形状上系统性出错，没有第二个独立实现能发现。

两条路径的契约（`tests/test_engines.py` 钉住）：

- 各自独立求出焊接 `f: S¹ → S¹`，不共享任何求解代码；
- 结果只允许相差**源圆旋转**（`hbs.weld_compare` 度量，N=600 时 `rms(δ−S) ≈ 0.1°`）；
- 焊接之后的一切（M1/M2 之后的 FEM 延拓、μ、I₂）**完全共用** `_pipeline`。

## 两个引擎

| 引擎 | 实现 | 特点 |
|---|---|---|
| `"angular"` | 边界层对数单层势 → 内外调和测度密度 → 圆相位 | 光滑曲线精度高；要求原始内外密度严格为正 |
| `"zipper"` | Bishop/Marshall Zipper（Möbius + 开方逐步收缩 slit） | 只依赖点序列，对不规则采样更宽容；对高对称形状有结构性退化 |

**互不替补**：任一引擎在自己的输入阶梯上全部失败时抛错，不会静默换到另一个。
需要另一条路径请显式改 `method`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d

from hbs.conformal_welding import (
    ConformalWelding,
    ConformalWeldingNumericalError,
    zipper_welding_prepared,
)


@dataclass(frozen=True)
class WeldingResult:
    """一个引擎给出的焊接及其诊断信息。

    `welding` 是 `angular_welding` 的字典形式或已构造好的 `ConformalWelding`；
    `rung` 非 `"raw"` 表示原始输入失败、用了降级的平滑重采样，调用方应告知用户。
    """

    welding: ConformalWelding
    rung: str
    engine: str


def _rungs(bound: np.ndarray) -> "list[tuple[str, np.ndarray]]":
    """失败时的输入阶梯：原始 → 弧长重采样 → 逐级高斯平滑。

    像素阶梯轮廓、噪声轮廓会让角速度解出现负的调和测度密度，直接失败。这几级是
    **降级**而非等价结果：救回的形状保真明显低于原始输入成功时，调用方会把
    `rung` 报出去。
    """
    from hbs.hbs import _resample_boundary

    resampled = _resample_boundary(bound, 500)
    yield "raw", bound
    yield "resample500", resampled
    for sigma in (2.0, 4.0):
        yield (f"resample500+s{sigma:g}",
               gaussian_filter1d(resampled, sigma, axis=0, mode="wrap"))


def _first_success(
    engine: str,
    bound: np.ndarray,
    attempt: Callable[[np.ndarray], ConformalWelding],
) -> WeldingResult:
    """按输入阶梯逐级尝试一个引擎，返回第一个成功的结果。

    任一引擎都**只用自己的阶梯**，失败时抛 `RuntimeError`，不换引擎。
    """
    last: Exception | None = None
    for rung, candidate in _rungs(bound):
        try:
            return WeldingResult(attempt(candidate), rung, engine)
        except (RuntimeError, ValueError, ConformalWeldingNumericalError) as exc:
            last = exc
    raise RuntimeError(
        f'{engine} 引擎在原始输入与全部 {len(tuple(_rungs(bound))) - 1} 级降级重试上'
        f'均失败（最后错误：{type(last).__name__}: {last}）。'
        "该引擎不适合这条边界；换 method 是**显式**选择，不会自动发生。"
    ) from last


def angular(bound: np.ndarray, circle_num: int) -> WeldingResult:
    """角速度引擎：边界层对数单层势求解。"""
    from hbs.angular import angular_welding

    return _first_success(
        "angular", bound,
        lambda candidate: ConformalWelding.from_angular(
            angular_welding(candidate, circle_num=circle_num)
        ),
    )


def zipper(bound: np.ndarray, circle_num: int) -> WeldingResult:
    """Zipper 引擎：Bishop/Marshall zipper 算法。"""
    return _first_success(
        "zipper", bound,
        lambda candidate: zipper_welding_prepared(candidate),
    )


#: 引擎注册表。新增引擎只需在这里登记，`get_hbs` / `get_conformal_welding` 自动可用。
ENGINES: dict[str, Callable[[np.ndarray, int], WeldingResult]] = {
    "angular": angular,
    "zipper": zipper,
}
