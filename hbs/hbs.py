from __future__ import annotations

from dataclasses import replace

import numpy as np

from hbs._engines import ENGINES
from hbs._pipeline import (
    DEFAULT_CIRCLE,
    DEFAULT_DENSITY,
    DEFAULT_NORMALIZE,
    HBSResult,
    build_hbs,
    prepare_boundary,
    resolve_disk,
)
from hbs.conformal_welding import ConformalWelding
from hbs.mesh import DiskMesh
from hbs.qc import lsqc_solver
from hbs.utils.cast import to_complex, to_real
from hbs.utils.geodesic_welding import geodesic_welding

__all__ = ["get_hbs", "reconstruct_from_hbs"]


def get_hbs(
    bound: np.ndarray[np.floating],
    circle_point_num: int = DEFAULT_CIRCLE,
    density: float = DEFAULT_DENSITY,
    disk: DiskMesh = None,
    method: str = "angular",
    normalize: str = DEFAULT_NORMALIZE,
    validate_reconstruction: bool = False,
) -> tuple[
    np.ndarray[np.complexfloating],
    np.ndarray[np.floating],
    ConformalWelding,
    DiskMesh,
]:
    """由边界点求 HBS（Beltrami 系数，定义在三角面心）。

    ## 管线

        边界 → 定向/锚点/弧长重采样 → 引擎求焊接 → FEM 调和延拓 → μ → 源圆旋转规范

    只有**引擎**按 `method` 切换，之后完全共用 `hbs._pipeline`。

    ## 两个引擎

    | `method` | 算法 | 特点 |
    |---|---|---|
    | `"angular"`（默认） | 边界层对数单层势 | 光滑曲线精度高；要求内外调和测度密度严格为正 |
    | `"zipper"` | Bishop/Marshall zipper | 只依赖点序列，对不规则采样宽容；高对称形状有结构性退化 |

    **两套实现互不替补**：任一引擎在自己的输入阶梯（原始 → 重采样 → 逐级平滑）上
    全部失败时抛 `RuntimeError`，不会静默换到另一条。要换引擎请显式改 `method`。
    两条路径的焊接只相差源圆旋转（`hbs.weld_compare` 度量，N=600 时
    `rms(δ−S) ≈ 0.1°`），因此 μ 在扣除该旋转后一致到离散误差。

    ## 参数

    :param `bound`: `n x 2` 边界点（任意绕向，内部统一翻到顺时针）
    :param `circle_point_num`: 源圆采样点数，同时决定重建分辨率上限
    :param `density`: 盘网格密度；与 `circle_point_num` 一起定位缓存的网格
    :param `disk`: 显式 `DiskMesh`；给定则 `density` 与 `circle_point_num` 只影响
        焊接的采样点数，网格本身原样使用
    :param `method`: `"angular"`（默认）或 `"zipper"`，见上表
    :param `normalize`: 源圆旋转规范，见下
    :param `validate_reconstruction`: 打开「重建合法性门 + 一次性粗网格回退」，
        见下节；默认关闭
    :return:
        `hbs`：`disk.face_num` 个复数，Beltrami 系数
        `field`：`disk.vert_num x 2` 实数组，调和延拓场
        `welding`：实际使用的 `ConformalWelding`
        `disk`：实际使用的 `DiskMesh`（规范后会整体旋转，**不是**传入的那个）
        （`method`、降级落点与规范诊断量另见 `get_hbs_result`）

    ## `validate_reconstruction`

    近临界输入（大量面片 `|μ| → 1`）会让重建塌缩成 `circle_num` 个完全相同的
    `1.0`——无异常、有限、但不是曲线。打开本开关后：

    1. 在 `(density, circle_point_num)` 上求解并试重建；
    2. 不合法则改用 `circle_point_num // 2`（下限 64）重解一次；
    3. 两级都不合法抛 `RuntimeError`，**不返回未校验的结果**。

    传入 `disk` 时不做门与回退（显式网格由调用方负责），与库外 1.2.0 角速度
    路径的语义一致。代价是每次调用多花一次完整重建（LSQC 求解）。

    ## `normalize`

    | 值 | 规范 | 说明 |
    |---|---|---|
    | `"auto"`（默认） | I₂ 可信时用 I₂，否则退到 M3+M4 | 两套互补，覆盖全部形状 |
    | `"i2"` | 只用 I₂ 单锚 | 无分支、连续；近对称形状方向不可信 |
    | `"m34"` | 论文 (5.8)+(5.9) 闭式两分支 | I₂ 近零时的替代 |
    | `"none"` | 不施加 | 引擎原样输出 |

    `"auto"` 的切换是不连续的（跨门限的相邻形状走不同规范，μ 相差一个旋转），
    诊断量在 `HBSResult.gauge` 里。

    :raises ValueError: `method` / `normalize` 非法，或边界退化
    :raises RuntimeError: 该引擎在全部输入阶梯上失败
    """
    result = get_hbs_result(
        bound,
        circle_point_num=circle_point_num,
        density=density,
        disk=disk,
        method=method,
        normalize=normalize,
        validate_reconstruction=validate_reconstruction,
    )
    return result.mu, result.field, result.welding, result.disk


def get_hbs_result(
    bound: np.ndarray[np.floating],
    circle_point_num: int = DEFAULT_CIRCLE,
    density: float = DEFAULT_DENSITY,
    disk: DiskMesh = None,
    method: str = "angular",
    normalize: str = DEFAULT_NORMALIZE,
    validate_reconstruction: bool = False,
) -> HBSResult:
    """`get_hbs` 的完整版本，额外交回引擎名、降级落点与规范诊断量。

    `HBSResult.rung` 非 `"raw"` 时表示原始输入失败、用了降级的平滑重采样：
    结果可用但保真明显下降，调用方应告知用户。

    `HBSResult.guard_rung` 非 `None` 表示开了 `validate_reconstruction` 且默认网格
    的重建不合法、换用了更粗的源圆：值形如 `"fallback:256"`，`disk.circle_num`
    与之相同。这时结果可用，但重建分辨率上限降到回退网格。

    `HBSResult.gauge` 是 `gauges.GaugeChoice`：`kind` 报实际用了哪套规范，
    `confidence_i2` / `confidence_m3` 报两个泛函的可信度，`branch_margin` 报
    M3+M4 离分支翻转还有多远。`kind == "none"` 且 `phase is None` 表示两套都不可信、
    没有施加旋转——这时 μ 的方位没有规范意义，下游不应跨形状比较相位。
    """
    if method not in ENGINES:
        raise ValueError(
            f"method must be one of {sorted(ENGINES)}, got {method!r}"
        )
    _validate_boundary(bound)

    result = _solve_hbs(
        bound, circle_point_num, density, disk, method, normalize
    )
    if not validate_reconstruction or disk is not None:
        return result

    from hbs._guard import MIN_FALLBACK_CIRCLE, reconstruction_is_valid

    if reconstruction_is_valid(result.mu, result.disk):
        return result

    fallback_circle = max(MIN_FALLBACK_CIRCLE, int(circle_point_num) // 2)
    if fallback_circle >= int(circle_point_num):
        raise RuntimeError(
            f"circle_point_num={circle_point_num} 已到回退下限"
            f"（{MIN_FALLBACK_CIRCLE}），无法再降级；请改用更粗网格或检查输入"
        )
    retry = _solve_hbs(
        bound, fallback_circle, density, None, method, normalize
    )
    if reconstruction_is_valid(retry.mu, retry.disk):
        return replace(retry, guard_rung=f"fallback:{retry.disk.circle_num}")
    raise RuntimeError(
        f"近临界 |μ|→1：重建边界在 {circle_point_num} 与 {fallback_circle} "
        "两个源圆上都退化为一点，拒绝返回非曲线"
    )


def _solve_hbs(
    bound: np.ndarray,
    circle_point_num: int,
    density: float,
    disk: DiskMesh | None,
    method: str,
    normalize: str,
) -> HBSResult:
    """一次完整的 焊接 → FEM 延拓 → μ → 规范，不含合法性门。"""
    prepared = prepare_boundary(np.asarray(bound, float), circle_point_num)
    welding_result = ENGINES[method](prepared, circle_point_num)
    disk, fem = resolve_disk(disk, density, circle_point_num)
    mu, field, welding, disk, gauge = build_hbs(
        welding_result.welding, disk, fem, circle_point_num, normalize
    )
    return HBSResult(
        mu=mu,
        field=field,
        welding=welding,
        disk=disk,
        engine=welding_result.engine,
        rung=welding_result.rung,
        gauge=gauge,
    )


def reconstruct_from_hbs(
    hbs: np.ndarray[np.complexfloating],
    disk: DiskMesh,
    eps: float = 0.0,
):
    """由 HBS 重建形状。

    :param `hbs`: `disk.face_num` 个复数，Beltrami 系数
    :param `disk`: `DiskMesh`（通常是 `get_hbs` 返回的那个）
    :param `eps`: LSQC 稳定性正则化（尖角处 `|μ|→1` 病态，>0 改善重建）
    :return:
        `bound_points`: `disk.circle_num x 2`，边界点
        `in_points`: `disk.in_vert_num x 2`，内部点
        `out_points`: `disk.out_vert_num x 2`，外部点
        `mapping`: `disk.vert_num x 2`，解出的映射顶点坐标
    """
    assert isinstance(disk, DiskMesh), "`disk` must be DiskMesh object"
    assert isinstance(hbs, np.ndarray) and np.issubdtype(
        hbs.dtype, np.complexfloating
    ), "`hbs` must be complex array"
    assert hbs.ndim == 1 and hbs.shape[0] == disk.face_num, (
        "the length of `hbs` is `n` and `n` is the number of faces"
    )

    landmark, target = _landmarks(disk)
    mapping = lsqc_solver(hbs, landmark, target, disk, eps)
    bound_points = mapping[: disk.circle_num]

    # 内部点必须从**边界结束处**切到外点开始处；`geodesic_welding` 的第一个参数
    # 也要带上前 `circle_num` 个边界点（它按 `[boundary; inner]` 的整体顺序工作）。
    inner = mapping[disk.circle_num : disk.circle_num + disk.in_vert_num]
    outer = mapping[disk.circle_num + disk.in_vert_num :]
    try:
        welded_in, out_points = geodesic_welding(
            to_complex(np.vstack([bound_points, inner])),
            to_complex(outer),
            to_complex(bound_points),
            to_complex(disk.circle),
        )
    except RuntimeError:
        welded = False
    else:
        welded_in, out_points = to_real(welded_in), to_real(out_points)
        welded = (
            np.all(np.isfinite(welded_in))
            and np.all(np.isfinite(out_points))
            and _weld_not_degenerate(bound_points, welded_in, disk)
        )

    if not welded:
        # 焊接数值退化（尖角/病态 seam/非规范代表）→ 降级为纯 LSQC 输出，不抛错。
        return bound_points, inner, outer, mapping
    # 焊接成功时边界换成焊接后的那条：`welded_in` 前 `circle_num` 个是新边界。
    return (welded_in[: disk.circle_num], welded_in[disk.circle_num :],
            out_points, mapping)


def _landmarks(disk: DiskMesh) -> tuple[np.ndarray, np.ndarray]:
    """LSQC 的钉点：落在 (1,0) 与 (0,0) 的顶点，钉到自身位置。

    网格被 I₂ 整体旋转过时这两个坐标已不在原处，故 `gauges.rotate_disk` 把原
    索引记在 `disk.landmark` 上；目标取这些顶点自己的坐标，等价于"钉到自身"
    （旋转是保角变换）。
    """
    stored = getattr(disk, "landmark", None)
    if stored is not None:
        landmark = np.asarray(stored, dtype=int)
    else:
        one = np.all(disk.vert == np.array([1.0, 0.0]), axis=1)
        zero = np.all(disk.vert == np.array([0.0, 0.0]), axis=1)
        landmark = np.where(one | zero)[0]
    return landmark, disk.vert[landmark].astype(np.float64)


def _weld_not_degenerate(
    bound_points: np.ndarray, welded_in_points: np.ndarray, disk: DiskMesh
) -> bool:
    """焊接退化检测：焊接后边界面积与 LSQC 边界面积之比过小（< 1e-3）。

    焊接可能输出有限但几何错误的点（如全部坍缩到 landmark），`isfinite` 抓不到。
    """
    def area(z):
        z = np.asarray(z)
        if z.ndim == 2:
            z = z[:, 0] + 1j * z[:, 1]
        closed = np.concatenate([z, z[:1]])
        return float(0.5 * np.abs(np.imag(np.conj(closed[:-1]) * closed[1:])).sum())

    a_lsqc = area(bound_points)
    return a_lsqc > 0 and area(welded_in_points[: disk.circle_num]) / a_lsqc > 1e-3


def _turning_angles(z: np.ndarray) -> np.ndarray:
    """闭合多边形每个顶点的有符号转角 `arg((z_{i+1}-z_i)/(z_i-z_{i-1}))`。

    旋转、平移、缩放不变；反转（改变绕行方向）只翻符号。零长度边的转角记为 0。
    """
    a = z - np.roll(z, 1)
    b = np.roll(z, -1) - z
    num = np.imag(np.conj(a) * b)
    den = np.real(np.conj(a) * b)
    return np.arctan2(num, den)


def _canonical_start_index(bound: np.ndarray, tol: float = 1e-9) -> int:
    """由点集几何决定的起点：离质心最远的顶点，按局部转角打破并列。

    为什么需要它：`_resample_boundary` 把弧长网格的原点钉在 `bound[0]` 上，输入
    起点一挪，重采样的**点集**就变了（不是循环移位），于是 μ 随 roll 变化。取一个
    只依赖点集的锚点后，预处理成为点集的函数，roll 变成纯粹的循环移位。

    主键 `|z - mean(z)|` 对平移/缩放/旋转/翻转协变；并列时（星形/多尖形状的等距
    尖端）用**局部转角**打破——它同样是旋转不变量，不能用 `argmax Re` 这类依赖
    当前坐标框架的量。若两者都并列（形状确有对称性），任取其一给出的重采样点集
    相同，取最小索引即可（确定性）。

    顺序约定：调用方须**先**做定向规范化（翻到顺时针）再调本函数，反转输入才会
    落到同一个锚点（转角符号随绕行方向翻转）。
    """
    z = np.asarray(bound, dtype=complex)
    if z.size < 3:
        return 0
    z = z[:, 0] + 1j * z[:, 1] if z.ndim == 2 else z
    d = np.abs(z - z.mean())
    dmax = float(d.max())
    if not np.isfinite(dmax) or dmax <= np.finfo(float).eps * max(1.0, float(np.abs(z).max())):
        return 0
    cand = np.flatnonzero(d >= dmax * (1.0 - tol))
    if cand.size == 1:
        return int(cand[0])
    turn = np.abs(_turning_angles(z))
    sub = cand[turn[cand] >= turn[cand].max() * (1.0 - tol)]
    return int(sub[0])


def _validate_boundary(bound: np.ndarray) -> None:
    """在数值焊接之前拒绝非法闭合曲线。"""
    if not isinstance(bound, np.ndarray) or bound.ndim != 2 or bound.shape[1] != 2:
        raise ValueError("boundary must be a finite n x 2 array")
    if bound.shape[0] < 3 or not np.issubdtype(bound.dtype, np.number):
        raise ValueError("boundary must contain at least three numeric points")
    if not np.all(np.isfinite(bound)) or np.unique(bound, axis=0).shape[0] < 3:
        raise ValueError("boundary must contain at least three distinct finite points")
    closed = np.vstack([bound, bound[:1]])
    perimeter = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
    if not np.isfinite(perimeter) or perimeter <= np.finfo(float).eps:
        raise ValueError("boundary must have positive perimeter")
    if abs(_signed_area(bound)) <= np.finfo(float).eps * perimeter**2:
        raise ValueError("boundary must enclose non-zero area")


def _signed_area(bound: np.ndarray) -> float:
    """多边形带符号面积（shoelace）。正 = 逆时针（math 约定），负 = 顺时针。"""
    x = bound[:, 0]
    y = bound[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _resample_boundary(bound: np.ndarray, num: int) -> np.ndarray:
    """按弧长均匀重采样闭合多边形到 `num` 点。"""
    z = bound[:, 0] + 1j * bound[:, 1]
    z = np.concatenate([z, z[:1]])
    d = np.abs(np.diff(z))
    s = np.concatenate([[0], np.cumsum(d)])
    s = s / s[-1]
    z2 = np.interp(np.linspace(0, 1, num, endpoint=False), s, z, period=1.0)
    return np.stack([z2.real, z2.imag], 1)
