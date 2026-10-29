"""焊接 → HBS 的**唯一**后端。两条引擎共用这一段，不重复实现。

    bound
      → validate / prepare（定向 → 几何锚点 → 弧长重采样）
      → <engine>            （`hbs._engines`，两套独立实现）
      → ConformalWelding
      → linear_interp        （重采样到源圆的均匀网格）
      → fem_extend           （P1 弱拉普拉斯求解，磁盘缓存的 A/B）
      → μ = H_z̄ / H_z
      → 源圆旋转规范          （`hbs.gauges.select_gauge`，坐标与相位同时转）

把后两端（延拓、μ、规范）放在一个模块里，是因为它们**不应**有第二份实现：历史上
angular 与 zipper 各写一套，同一个规范化的语义必然漂移，连返回的 `disk` 顶点都对不上。

## 规范的两套机制

`hbs.gauges` 提供 I₂ 与论文 M3+M4 两套源圆旋转规范，`normalize` 选择：

| `normalize` | 规范 | 何时可靠 |
|---|---|---|
| `"default"`（默认） | 无分支锚 `C₀ = M₁·conj(M₃)` | 非旋转对称形状；`C₀` 归零即报未定 |
| `"auto"` | I₂ 可信度够高用 I₂，否则退到 M3+M4 | 全部形状：两套互补（已弃用，等价于旧默认） |
| `"i2"` | 只用 I₂ | 非近对称形状；`abs(I₂)/Σabs(term)` 高 |
| `"m34"` | 只用论文 (5.8)+(5.9) | I₂ 近零但 M₃ 不近零 |
| `"c0"` | 无分支锚 `C₀ = M₁·conj(M₃)` | 非旋转对称形状；`C₀` 归零即报未定 |
| `"none"` | 不施加 | 需要引擎原样输出时 |

`"auto"` 的切换门是 `gauges.AUTO_I2_CONFIDENCE`。切换本身是**不连续**的：跨过门限的
相邻形状会走不同的规范，输出的 μ 相差一个旋转。这是刻意的取舍——近零锚不可靠时
宁可换一套有独立方向信息的锚，而不是在噪声方向上转（见 `研究/C1与I2旋转锚的互补条件.md`）。
`HBSResult.gauge` 把选了哪一套、可信度多少、M3+M4 的分支裕度都报出来，调用方可以
据此判断这一帧是否该采信。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import hbs.cache as cache
import hbs.gauges as gauges
from hbs.conformal_welding import ConformalWelding
from hbs.extension import fem_extend
from hbs.mesh import DiskMesh
from hbs.qc import get_beltrami_coefficient

#: 默认网格参数。两条引擎、两个公开入口都用这一对。
DEFAULT_DENSITY = 0.01
DEFAULT_CIRCLE = 500

#: μ 源圆旋转规范的合法取值，语义见模块文档与 `hbs.gauges.GAUGES`。
NORMALIZATIONS = gauges.GAUGES

#: 默认规范：无分支锚 C₀（`default` = `c0`）。
DEFAULT_NORMALIZE = "default"


@dataclass(frozen=True)
class HBSResult:
    """`get_hbs` 的完整输出。"""

    mu: np.ndarray
    field: np.ndarray
    welding: ConformalWelding
    disk: DiskMesh
    engine: str
    rung: str
    gauge: gauges.GaugeChoice
    """实际施加的规范与诊断量（`kind` / `phase` / 两个可信度 / 分支裕度）。"""

    guard_rung: str | None = None
    """重建合法性门的落点。`None` 表示未开开关或默认网格已合法；
    `"fallback:<circle>"` 表示默认网格重建退化、换用了更粗的源圆，
    此时 `disk.circle_num` 与它报的值一致。"""


def prepare_boundary(bound: np.ndarray, resample_to: int) -> np.ndarray:
    """定向 → 几何锚点 → 弧长重采样。

    三步都必须在焊接之前完成，且**只能有一份实现**：`get_hbs` 与
    `get_conformal_welding` 是两个公开入口，若其中一个把原始 `bound` 直接喂给
    引擎，同一形状在两个入口上会给出不同的 `(x, y)`。

    1. **定向**：调和延拓只在顺时针边界（signed area < 0）上产生 `|μ| < 1` 的场；
       逆时针输入会让 `|μ|` 全盘饱和到 1。图轮廓（cv2 在图像坐标下输出顺时针）
       与合成形状方向不一致，统一翻到顺时针。
    2. **锚点**：`resample` 把弧长网格的原点钉在 `bound[0]`，输入起点一挪，
       重采样的**点集**就变了（不是循环移位）。改用由点集几何决定的锚点后，
       起点 roll 变成纯循环移位，μ 逐位不变。
    3. **重采样**：zipper 对非均匀采样（cv2 轮廓、密集采样、顶点落坐标轴）产生
       NaN，统一按弧长重采样规避。
    """
    from hbs.hbs import _canonical_start_index, _resample_boundary, _signed_area

    if _signed_area(bound) > 0:
        bound = bound[::-1]
    bound = np.roll(bound, -_canonical_start_index(bound), axis=0)
    return _resample_boundary(bound, resample_to)


def resolve_disk(
    disk: DiskMesh | None, density: float, circle_num: int
) -> tuple[DiskMesh, tuple]:
    """取盘与它的 FEM 弱拉普拉斯块 `(A, B)`。

    调用方自带的盘不带 FEM 块，就地构造（同网格上仍可复用 `cache`）。
    """
    if disk is None:
        return cache.load_unit_disk(density, circle_num)
    return disk, cache._fem_operator(disk)


def build_hbs(
    welding: ConformalWelding,
    disk: DiskMesh,
    fem: tuple,
    circle_num: int,
    normalize: str = DEFAULT_NORMALIZE,
) -> tuple[np.ndarray, np.ndarray, ConformalWelding, DiskMesh, gauges.GaugeChoice]:
    """焊接 → μ。两引擎共用，唯一实现。

    返回 `(mu, field, welding, disk, gauge)`；`disk` 在施加规范后是**新的**
    `DiskMesh`（顶点整体旋转），调用方必须使用返回值而不是入参。
    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}")

    welding.linear_interp(circle_num)
    field, _ = fem_extend(disk, np.asarray(welding.y, complex), fem=fem)
    mu = get_beltrami_coefficient(field, disk)

    choice = gauges.select_gauge(mu, disk, normalize)
    if choice.phase is None:
        # 两套规范都不可信（或 normalize="none"）：原样返回，不在噪声方向上旋转。
        return mu, field, welding, disk, choice

    # 规范：**坐标与相位同时转**（`z' = e^{-iθ}z`、`μ' = e^{-2iθ}μ`）。
    # 只乘相位是错的：那等价于假设 μ 在旋转下不变（只有圆才成立），
    # 实测会让重建残差从 1e-6 恶化到 1e-3~5e-2。
    mu, scale, choice = gauges.apply_gauge(mu, disk, choice)
    disk = gauges.rotate_disk(disk, scale)
    # 源圆角与顶点同转；目标圆角不动（映射没变，只是源参数化旋转了）。
    welding.rotate_x(choice.phase)
    return mu, field, welding, disk, choice


@dataclass(frozen=True)
class EngineAgreement:
    """两条引擎的 μ 差异，**已扣除源圆旋转**。"""

    welding_shift: float
    """对齐所需源圆旋转（弧度）。"""

    welding_rms: float
    """焊接对齐后的 `rms(δ−S)`（弧度）。"""

    mu_max: float
    mu_rms: float
    """μ 的差异：`max` 受边界层主导，判定用 `rms`。"""


def compare_engines(
    first: ConformalWelding,
    second: ConformalWelding,
    disk: DiskMesh,
    fem: tuple,
    circle_num: int,
) -> EngineAgreement:
    """比较两条引擎的 μ。**前提是对齐源圆参数**。

    两条焊接只相差源圆旋转 `T*`（`hbs.weld_compare`）。若直接拿各自的 `y` 送进
    FEM，比的是两个不同的源参数化：实测同一形状会报出 `μ` 相对差 0.80，而把
    `T*` 扣掉后只有 0.006。这个坑很容易踩，所以比较逻辑收在这里，不让调用方自己写。

    `mu_max` 由边界层（`r > 0.97`）主导：`μ = H_z̄/H_z` 是导数之比，边界附近对
    输入的条件数放大。判定两引擎是否一致应当看 `mu_rms`。
    """
    from hbs.qc import get_beltrami_coefficient
    from hbs.weld_compare import compare_conformal_welding, welding_curve

    agreement = compare_conformal_welding(first, second)

    def mu_of(boundary_values):
        field, _ = fem_extend(
            disk, np.asarray(boundary_values, complex), fem=fem
        )
        return get_beltrami_coefficient(field, disk)

    # 把 first 的焊接搬到 second 的源圆参数上：y(θ) = f₁(θ+T*) − f₁(T*)
    curve = welding_curve(first)
    grid = 2 * np.pi * np.arange(circle_num) / circle_num
    aligned = np.exp(
        1j * (curve(grid + agreement.shift) - curve(np.array([agreement.shift]))[0])
    )
    mu_first = mu_of(aligned)
    mu_second = mu_of(np.asarray(second.y, complex))

    delta = np.abs(mu_first - mu_second)
    return EngineAgreement(
        welding_shift=float(agreement.shift),
        welding_rms=float(agreement.rms_residual),
        mu_max=float(delta.max() / max(np.abs(mu_second).max(), 1e-12)),
        mu_rms=float(
            np.sqrt(np.mean(delta**2))
            / max(np.sqrt(np.mean(np.abs(mu_second) ** 2)), 1e-12)
        ),
    )
