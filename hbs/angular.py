"""2.0.0：角速度 shape→HBS 求解器，`get_hbs(method="angular")` 的实现。

1.1.0 的 `get_hbs` 只有 Zipper+30 轮归一化路径（尖角/对称系已知不收敛）。
本模块提供角速度路径；`method="zipper"` 保留 1.1.0 行为。

    bound(n×2 点数组)
      → hbs._angular.direct.compute_direct_hbs_from_boundary  (角速度求解)
      → 调和延拓落场 (hbs._angular.joint.JointHBSProblem + 两次 Laplace 回代)
      → mu[face] (Beltrami 系数, 定义在面片上)

两个引擎模块随包 vendored 在 `hbs/_angular/`，只用 numpy/scipy，包外无依赖。
`reconstruct_from_hbs(mu, disk)` 原样消费本模块产出的 mu。
密度守门语义与 `upgraded_hbs.compute_upgraded_hbs` 一致：原始内外角速度
任一非正则直接抛错，不做密度修剪（角速度路径的 6 例失败即死于此门）。

网格选择（shape100-v1 100 例实测 [X]）：默认 `(0.04, 512)`。边界点数决定重建
分辨率上限，内部面片密度无关 —— `(0.02, 256)` 与 `(0.04, 256)` 逐位相同，
而 `(0.04, 512)` 在共同通过子集上 58:23 胜 `(0.04, 256)`（p=1.3e-4），
耗时只多 34%。Zipper 风格的 `(0.01, 500)`（60796 面片）比 `(0.04, 512)`
更差且慢 7 倍，属纯浪费。

近临界例外：`real_chopper-3` 在 circle≥320 时有 90 个面片 `|mu|>0.9999`，
LSQC 仍有界但 `geodesic_welding` 把所有外点压到 `|z|=1`。故默认路径带
**一次性回退**：默认网格重建不合法时，改用 `(0.04, 256)` 重算（质量仍取默认
网格；`_reconstruct_is_valid` 会顺带花掉一次重建）。

**回退结果必须再校验**（2026-10-24 修复）。此前回退直接返回、不再检查，
冻结 100 例里 `concave_quad`、`real_Bone-3`、`real_hammer-3` 三例因此
**静默**拿到 512 个完全相同的 `1.0`（无异常、输出有限、`extent=0`）；现在
两级都不过时抛 `RuntimeError`，不再返回非曲线。换网格不能普适修复这三例：
`n∈{1024,2048} × circle∈{512,1024}` 均为哨兵，只有 `concave_quad` 在
`n=2048` 下恢复 [X]。`eps>0` 也能救 chopper-3，但会让 apple/star6 精度掉
约一半，故不默认开启。
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from hbs.mesh import DiskMesh, get_unit_disk

_DEFAULT_DENSITY = 0.04
_DEFAULT_CIRCLE = 512
_FALLBACK_CIRCLE = 256

_mesh_cache: dict = {}


def _mesh_for(circle: int) -> DiskMesh:
    mesh = _mesh_cache.get(circle)
    if mesh is None:
        mesh = get_unit_disk(_DEFAULT_DENSITY, circle)
        _mesh_cache[circle] = mesh
    return mesh


def _default_mesh() -> DiskMesh:
    return _mesh_for(_DEFAULT_CIRCLE)


class _SplineShape:
    """最小 shape 适配器：只提供 curve/derivative/sample_t。"""

    def __init__(self, bound: np.ndarray):
        z = np.asarray(bound[:, 0], float) + 1j * np.asarray(bound[:, 1], float)
        if np.abs(z[0] - z[-1]) < 1e-13:
            z = z[:-1]
        area = np.sum(z.real * np.roll(z.imag, -1) - z.imag * np.roll(z.real, -1))
        if area < 0:
            z = z[::-1]
        edges = np.abs(np.diff(np.r_[z, z[0]]))
        if edges.min() <= 1e-14:
            raise ValueError("repeated or degenerate boundary points")
        s = np.r_[0.0, np.cumsum(edges)]
        t = 2 * np.pi * s / s[-1]
        self._x = CubicSpline(t, np.r_[z.real, z.real[0]], bc_type="periodic")
        self._y = CubicSpline(t, np.r_[z.imag, z.imag[0]], bc_type="periodic")
        self.sample_t = t[:-1]

    def curve(self, v):
        v = np.asarray(v, float)
        return self._x(v) + 1j * self._y(v)

    def derivative(self, v):
        v = np.asarray(v, float)
        return self._x(v, 1) + 1j * self._y(v, 1)


def angular_mu_from_boundary(
    bound: np.ndarray,
    n: int = 1024,
    disk: DiskMesh | None = None,
) -> tuple[np.ndarray, np.ndarray, DiskMesh, object, dict]:
    """角速度路径：bound → mu[face]。

    返回 5 元组 `(mu, field_real, disk, result, stats)`：

    - `mu`：`disk.face_num` 个复数，Beltrami 系数，直接喂 `reconstruct_from_hbs`；
    - `field_real`：FEM 场 `vert_num×2` 实数组（`get_hbs` 的 `he` 槽位）；
    - `disk`：实际使用的网格（可能不是 `(0.04, 512)`，见下）；
    - `result`：`DirectHBS` 求解记录；
    - `stats`：`min_jacobian` / `max_abs_mu` / `source_phase`。

    `disk=None` 时用默认网格 `(0.04, 512)`；若该网格重建出的边界不合法
    （近临界 |mu|→1 使焊接塌陷），自动回退到 `(0.04, 256)` 重算并返回粗网格的
    disk —— 调用方拿到的 mu/disk 始终自洽。**显式传入 `disk` 只禁用这层网格回退，
    不禁用下面的输入阶梯**：阶梯换的是输入曲线，不是网格。

    **输入阶梯（仅失败时）**：角速度求解要求边界光滑。像素阶梯轮廓、噪声轮廓
    会让原始内外角速度出现负值，直接失败。此时按
    `原始 → 弧长重采样500 → 重采样500+σ2 高斯 → 重采样500+σ4 高斯`
    逐级重试，第一个通过合法性门的结果即返回，并用 `stats["boundary_rung"]`
    / `stats["input_smoothed"]` 标记（非 `"raw"` 时同时发 `UserWarning`）。
    原始输入成功时阶梯不启动，结果与无阶梯完全一致。

    MPEG-7 抽样实测 [X]（真实 cv2 轮廓；可用性 = 重建有限非单点 且 尺度/方向无关
    形状保真 < 0.30）：阶梯设计探针 **95 例 raw 68 → 阶梯 72**（zipper 69）；
    装成 wheel 后**用本文件实现**复核 55 例 **raw 37 → 阶梯 39**（zipper 38）。
    被救回例的保真 0.049–0.157，明显劣于原始输入成功时的中位 0.0019
    ——「可用但退化」，故必须标记，不得当作等价结果。剩余约 20%–23% 两引擎都拒绝。

    **回退结果同样过合法性门。** 两级都不合法时抛 `RuntimeError`，不返回
    退化边界：`geodesic_welding` 在近临界时会静默把整条边界写成精确的 `1.0`，
    若不校验，调用方拿到的是"有限、无异常、但是一个点"的假曲线。

    :raises RuntimeError: 全部阶梯与网格回退都无法给出合法重建边界。
    """
    bound = np.asarray(bound, float)
    if bound.ndim != 2 or bound.shape[1] != 2 or len(bound) < 8:
        raise ValueError("bound must be an n-by-2 smooth closed curve sample")

    explicit = disk is not None
    last_exc = None
    for rung, candidate in _boundary_rungs(bound):
        try:
            out = (
                _mu_on_mesh(candidate, n, disk)
                if explicit
                else _solve_with_mesh_fallback(candidate, n)
            )
        except (RuntimeError, ValueError) as exc:
            last_exc = exc
            continue
        if rung != "raw":
            warnings.warn(
                f"angular solve failed on the raw boundary; used fallback rung "
                f"{rung!r} (the returned shape is usable but degraded)",
                UserWarning,
                stacklevel=2,
            )
            out[4]["boundary_rung"] = rung
            out[4]["input_smoothed"] = True
        else:
            out[4]["boundary_rung"] = "raw"
            out[4]["input_smoothed"] = False
        return out

    raise RuntimeError(
        "角速度路径在原始输入与全部 3 级平滑重试上均失败"
        f"（最后错误：{type(last_exc).__name__}: {last_exc}）。"
        "该边界不是光滑 Jordan 曲线，或近临界 |mu|->1；请先平滑/重采样输入，"
        '或改用 method="zipper"'
    ) from last_exc


def _boundary_rungs(bound: np.ndarray):
    """failure-only 输入阶梯：(标签, 候选边界)，由原始到最强平滑。

    弧长重采样把像素阶梯轮廓变成均匀采样；高斯平滑压掉拐角处的负角速度。
    全部候选都保持闭合曲线形态，不改变调用方拿到的 disk。
    """
    from scipy.ndimage import gaussian_filter1d

    from hbs.hbs import _resample_boundary

    yield "raw", bound
    resampled = _resample_boundary(bound, 500)
    yield "resample500", resampled
    for sigma in (2.0, 4.0):
        yield (
            f"resample500+s{sigma:g}",
            gaussian_filter1d(resampled, sigma, axis=0, mode="wrap"),
        )


def _solve_with_mesh_fallback(
    bound: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, DiskMesh, object, dict]:
    """默认网格 → 合法性门 → 粗网格回退 → 再校验。"""
    out = _mu_on_mesh(bound, n, _default_mesh())
    if _reconstruct_is_valid(out[0], out[2]):
        return out

    # 默认网格不合法：回退一次（100 例中 4 例触发、1 例被救回）
    coarse = _mu_on_mesh(bound, n, _mesh_for(_FALLBACK_CIRCLE))
    if _reconstruct_is_valid(coarse[0], coarse[2]):
        return coarse
    raise RuntimeError(
        "近临界 |mu|→1：重建边界在默认网格与回退网格上均退化为一点"
        f"（circle={_DEFAULT_CIRCLE} 与 {_FALLBACK_CIRCLE}）；"
        "该输入无法由角速度路径给出可用边界，拒绝返回非曲线"
    )


def _reconstruct_is_valid(mu: np.ndarray, disk: DiskMesh) -> bool:
    """重建合法性门：边界必须有限、且不是一个点（`geodesic_welding` 的哨兵）。

    焊接塌陷时输出是 circle_num 个完全相同的 `1.0`：无异常、无 nan、`extent=0`。
    故这里只认"有非零展布"的边界；`extent` 同时覆盖常数与全部重合两种退化。
    """
    from hbs.hbs import reconstruct_from_hbs

    try:
        bound_rec, _, _, _ = reconstruct_from_hbs(np.asarray(mu, complex), disk)
    except Exception:
        return False
    z = bound_rec[:, 0] + 1j * bound_rec[:, 1]
    if z.shape != (disk.circle_num,) or not np.all(np.isfinite(z)):
        return False
    extent = float(np.max(np.abs(z - z.mean())))
    return extent > 1e-6


def _mu_on_mesh(
    bound: np.ndarray, n: int, disk: DiskMesh
) -> tuple[np.ndarray, np.ndarray, DiskMesh, object, dict]:
    """在指定网格上求解并落场（`angular_mu_from_boundary` 的实现体）。"""
    from hbs._angular import direct
    from hbs._angular.joint import JointHBSProblem

    result = direct.compute_direct_hbs_from_boundary(bound, n=n)
    if result.min_raw_internal_density <= 0 or result.min_raw_external_density <= 0:
        raise RuntimeError(
            "角速度基线需要原始内外角速度严格为正，不接受密度修剪"
        )

    shape = _SplineShape(bound)
    template = JointHBSProblem(shape, None, disk)
    t = 2 * np.pi * np.arange(len(result.alpha)) / len(result.alpha)
    alpha = CubicSpline(
        np.r_[t, 2 * np.pi], np.r_[result.alpha - t, 0.0], bc_type="periodic"
    )
    inv = PchipInterpolator(
        np.r_[result.beta, 2 * np.pi], np.r_[t, 2 * np.pi]
    )
    leading = (2 * np.pi / len(t)) * np.sum(
        shape.curve(t) * np.exp(-1j * result.beta) * result.rho_external
    )
    phase = -float(np.angle(leading))
    parameter = inv(np.mod(template.theta + phase, 2 * np.pi))
    boundary = np.exp(1j * (parameter + alpha(parameter)))
    values = np.empty(len(template.z), complex)
    values[template.bd] = boundary
    rhs = -template.B @ boundary
    values[template.ii] = template.source_factor.solve(
        rhs.real
    ) + 1j * template.source_factor.solve(rhs.imag)

    hz = disk.Dz @ values
    hc = disk.Dc @ values
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.asarray(hc / hz, complex).ravel()
    jac = np.abs(hz) ** 2 - np.abs(hc) ** 2
    field_real = np.stack([values.real, values.imag], axis=-1)
    stats = dict(
        min_jacobian=float(np.min(jac)),
        max_abs_mu=float(np.max(np.abs(mu[np.isfinite(mu)])))
        if np.any(np.isfinite(mu))
        else float("inf"),
        source_phase=float(phase),
    )
    return np.asarray(mu, complex), field_real, disk, result, stats
