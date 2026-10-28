"""角速度引擎：边界层对数单层势 → 共形焊接。

这是 `hbs._engines.angular` 的实现。**只负责焊接**：之后的调和延拓、μ、I₂ 规范
全部在 `hbs._pipeline`，两条引擎共用。

## 原理

光滑、正向的闭合 Jordan 曲线 `γ(t)` 上解两个第一类对数单层势方程，得到内外
调和测度密度 `ρ_in`、`ρ_out`，积分出源/目标圆相位 `α`、`β`。焊接是

    f: e^{iβ(t)} ↦ e^{iα(t)}

即目标圆角是**绝对角** `t + α(t)`（`α` 是相对增量，漏掉 `t` 会让 `y` 绕圆
0 圈而非 1 圈）。

## 规范化

求解器自带论文的两条规范化，不需要后处理：

- **M1**（式 5.11 / Algorithm 6.2）：`Σ_k Φ₁⁻¹(z_k) = 0`。`center_equations`
  对**全部 n 个离散点**求解（不是输入样本子集），残差实测 `1e-16`。
- **M2**（式 5.13 / Algorithm 6.3）：`Φ₂(∞) = ∞`。β 侧外域源角天然满足。

`f(1)=1` 与 `Φ₂'(∞) ≥ 0` **不施加**：前者只规范焊接的呈现方式、对 μ 无影响
（H 乘常数不改变 `H_z̄/H_z`，且 I₂ 会再转掉 T 的残差）；后者与 I₂ 是同一
转动自由度，同时施加会冲突。

## 局限

要求边界光滑，且原始内外调和测度密度严格为正——像素阶梯轮廓、噪声轮廓、尖角
会让密度出现负值，直接失败。这类输入的降级重试在 `hbs._engines._rungs`。
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from hbs.mesh import DiskMesh


class _SplineShape:
    """最小 shape 适配器：只提供 `curve` / `derivative` / `sample_t`。

    内部强制逆时针（`direct` 求解器要求正向参数化）；输入已被
    `hbs._pipeline.prepare_boundary` 统一翻成顺时针，所以这里会翻一次。
    """

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


def angular_welding(
    bound: np.ndarray, circle_num: int = 500, n: int = 1024
) -> dict:
    """求角速度焊接，返回 `ConformalWelding.from_angular` 需要的字典。

    :param `bound`: `n x 2` 光滑闭合曲线采样（会被 `_SplineShape` 翻成逆时针）
    :param `circle_num`: 源圆采样点数
    :param `n`: 边界层方程的离散点数（须为偶数），与 `circle_num` 无关
    :return: `init_x_angle` / `init_x` / `init_y` / `parameter` / `boundary`
    :raises RuntimeError: 原始内外调和测度密度出现非正值（不接受密度修剪）
    """
    from hbs._angular import direct

    result = direct.compute_direct_hbs_from_boundary(bound, n=n)
    if result.min_raw_internal_density <= 0 or result.min_raw_external_density <= 0:
        raise RuntimeError("角速度基线需要原始内外角速度严格为正，不接受密度修剪")

    shape = _SplineShape(bound)
    t = 2 * np.pi * np.arange(len(result.alpha)) / len(result.alpha)
    alpha = CubicSpline(
        np.r_[t, 2 * np.pi], np.r_[result.alpha - t, 0.0], bc_type="periodic"
    )
    inv = PchipInterpolator(np.r_[result.beta, 2 * np.pi], np.r_[t, 2 * np.pi])

    # 焊接 f: θx → θy 的两个角都是**抽象单位圆**上的角，与形状在平面上的朝向
    # 无关，所以这里不能掺入由 `shape.curve(t)` 算出的 `leading` 相位：那个相位
    # 承载形状朝向 θ（实测输入旋转 θ 时其幅角恰好增加 θ），一旦进入 `y`，FEM
    # 延拓的边界值就跟着转，μ 随输入旋转变化 1.52。
    #
    # 实测（200 点源角，旋转 0.7 rad）：
    #   phase = -arg(leading) → y 角差 mean −0.7000, std 5.7e-01   ❌
    #   phase = 0             → y 角差 mean −0.0000, std 1.4e-15   ✅
    theta = np.mod(np.arange(int(circle_num)) * 2 * np.pi / int(circle_num), 2 * np.pi)
    parameter = inv(theta)
    return {
        "init_x_angle": theta,
        "init_x": np.exp(1j * theta),
        "init_y": np.exp(1j * (parameter + alpha(parameter))),
        "parameter": parameter,
        "boundary": shape.curve(parameter),
    }


def angular_mu_from_boundary(
    bound: np.ndarray,
    n: int = 1024,
    disk: "DiskMesh | None" = None,
    normalize: str = "auto",
) -> tuple[np.ndarray, np.ndarray, "DiskMesh", object, dict]:
    """角速度路径的便捷入口：`bound → (mu, field, disk, result, stats)`。

    2.2.0 起这是 `hbs._pipeline` 的薄封装，与 `get_hbs(method="angular")` 走**同一
    段代码**（此前 `_mu_on_mesh` 是一条独立实现，同一规范化有两份，语义必然漂移）。

    保留这个签名是为了兼容旧调用方；新代码应直接用 `hbs.get_hbs` 或
    `hbs.get_hbs_result`（后者能拿到引擎名与降级落点）。

    :param `n`: 边界层离散点数（传给 `angular_welding`），默认 1024
    :param `disk`: 显式网格；省略时用 `(density=0.01, circle=500)` 的缓存网格
    :param `normalize`: `"auto"`（默认）/`"i2"`/`"m34"`/`"none"`，与 `get_hbs` 同义
    :raises RuntimeError: 全部输入阶梯失败
    """
    from hbs._pipeline import build_hbs, prepare_boundary, resolve_disk
    from hbs._engines import angular as _engine

    circle_num = disk.circle_num if disk is not None else 500
    prepared = prepare_boundary(np.asarray(bound, float), circle_num)
    result = _engine(prepared, circle_num)
    result.welding.linear_interp(circle_num)
    disk, fem = resolve_disk(disk, 0.01, circle_num)
    mu, field, welding, disk, gauge = build_hbs(
        result.welding, disk, fem, circle_num, normalize
    )
    return mu, field, disk, None, {
        "engine": result.engine,
        "rung": result.rung,
        "normalize": normalize,
        "welding": welding,
        "gauge": gauge,
    }
