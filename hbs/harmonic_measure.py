"""调和测度引擎：边界层对数单层势 → 共形焊接。

这是 `hbs._engines.harmonic_measure` 的实现。**只负责焊接**：之后的调和延拓、μ、I₂ 规范
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

## 局限与近角处理

要求边界光滑；像素阶梯轮廓、噪声轮廓仍可能失败。近角轮廓会让原始密度出现
离散化级负值（−1e-9～−1e-6）。`harmonic_measure_welding` 用配点阶梯
`n=1024→8192` 加密，接受判据是**两条同时**成立：负质量分数
`sum|min(q,0)|/sum|q| ≤ 1e-8`，且最深单点负值与正峰之比 ≤ 1e-6。峰比是分界线：
只有质量约束会把真正的几何退化（尖点、重噪声）当成伪影救回来。非首级或非严格正
的接受还须通过相邻级 β 差 ≤ 1e-2 rad 的收敛检查，`2n` 解达不到判据时该级按未收敛
拒绝。实际使用的 n、负质量/峰比与 β 收敛差随返回值报告。几何本身不被修改；几何
降级重试仍在 `hbs._engines._rungs`，且**配点阶梯只用于原始输入**，降级几何只跑
单级严格判据。
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from hbs.mesh import DiskMesh


class _SplineShape:
    """最小 shape 适配器：只提供 `curve` / `derivative` / `points`。

    内部强制逆时针（`direct` 求解器要求正向参数化）；输入已被
    `hbs._pipeline.prepare_boundary` 统一翻成顺时针，所以这里会翻一次。
    `points` 是翻转后的复向量，供极点搜索复用，避免重复拟合。
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
        self.points = z

    def curve(self, v):
        v = np.asarray(v, float)
        return self._x(v) + 1j * self._y(v)

    def derivative(self, v):
        v = np.asarray(v, float)
        return self._x(v, 1) + 1j * self._y(v, 1)


def harmonic_measure_welding(
    bound: np.ndarray,
    circle_num: int = 500,
    n: int = 1024,
    *,
    n_ladder: tuple[int, ...] | None = None,
    negative_mass_tolerance: float = 1e-8,
    negative_peak_tolerance: float = 1e-6,
    check_convergence: bool = True,
    angle_tolerance: float = 1e-2,
) -> dict:
    """求调和测度焊接，返回 `ConformalWelding.from_harmonic_measure` 需要的字典。

    近角轮廓的原始密度可能出现离散化级负值。负质量分数
    ``sum|min(q,0)|/sum|q|`` 随配点数约以 n^-2 下降；本函数按
    ``n_ladder`` 逐级加密，只有**两条**判据同时满足的那一级才被接受：

    1. 负质量分数 ≤ ``negative_mass_tolerance``；
    2. 最深单点负值与正峰之比 ≤ ``negative_peak_tolerance``。

    第二条是分界线：真几何退化（尖点 ``|γ'|=0``、σ 很大的噪声轮廓）的负值与
    密度本身同量级（实测噪声例 0.5–1.7、尖点例 0.19–0.59，且配点加密后**不复现
    消失**：5.2e-3 → 1.6e-2 → 0 → 5.3e-3），而离散化伪影的峰比随 n 掉到 1e-8–1e-12。
    只用质量会把前者当后者救回来，库内两支守卫测试正是这样失效的。

    非首级或非严格正的接受还须核实收敛：更细一级（``2n``）的解必须**自己达到
    判据**且 β 差 ≤ ``angle_tolerance``；``2n`` 解达不到判据时该级按未收敛拒绝，
    不再记一条 note 之后照常接受。

    :param `bound`: `n x 2` 光滑闭合曲线采样（会被 `_SplineShape` 翻成逆时针）
    :param `circle_num`: 源圆采样点数
    :param `n`: 单级配点数（`n_ladder` 为 None 时只用这一级）
    :param `n_ladder`: 候选配点数阶梯（升序使用）；每级都失败时抛错
    :param `negative_mass_tolerance`: 可接受的负质量分数上限
    :param `negative_peak_tolerance`: 可接受的「最深单点负值 / 正峰」上限
    :param `check_convergence`: 在非首级或非严格正接受时，用 2n 解检查 β 收敛
    :param `angle_tolerance`: 相邻两级的 β 最大差上限（弧度）。默认 1e-2，
        与 TBS 的边界相位预算 1.1e-2 rad 同量级；实测最难的近角输入
        （real_fork-3，1000 点，n=8192）为 6.2e-3，必须随记录报告。
    :return: `init_x_angle` / `init_x` / `init_y` / `parameter` / `boundary`，
        以及 `n`、`density_acceptance`、负质量/峰比与收敛诊断
    :raises RuntimeError: 全部配点级都未达到接受判据
    """
    from hbs._harmonic_measure import direct

    # 样条只在这里拟合一次：求解器的 `curve`/`derivative`、极点搜索与下游的
    # `shape.curve(parameter)` 共用同一个 `_SplineShape`。
    shape = _SplineShape(bound)
    pole = direct._interior_pole(shape.points)
    candidates = tuple(sorted({int(v) for v in
                               (n_ladder if n_ladder is not None else (n,))}))
    if not candidates or any(v < 8 or v % 2 for v in candidates):
        raise ValueError("n_ladder must contain even integers >= 8")

    diagnostics: list[dict] = []
    result = None
    accepted_n = None
    acceptance = None
    beta_convergence = None
    previous: tuple[int, object] | None = None

    def _beta_gap(n_a: int, beta_a, n_b: int, beta_b) -> float:
        grid_n = max(n_a, n_b)
        grid = 2 * np.pi * np.arange(grid_n) / grid_n
        a = np.interp(grid, 2 * np.pi * np.arange(n_a) / n_a, beta_a)
        b = np.interp(grid, 2 * np.pi * np.arange(n_b) / n_b, beta_b)
        return float(np.max(np.abs(a - b)))

    for candidate_n in candidates:
        try:
            trial = direct.compute_direct_hbs(
                shape.curve, shape.derivative, n=candidate_n, pole=pole,
                negative_mass_tolerance=negative_mass_tolerance,
                negative_peak_tolerance=negative_peak_tolerance,
            )
        except RuntimeError as exc:
            diagnostics.append({"n": candidate_n, "status": "rejected",
                                "error": str(exc)})
            continue
        strict = (trial.min_raw_internal_density >= 0
                  and trial.min_raw_external_density >= 0)
        record = {"n": candidate_n, **trial.density_diagnostics()}
        record.pop("density_regularization_mass", None)
        if check_convergence and (candidate_n != candidates[0] or not strict):
            reference = None
            reference_n = None
            if 2 * candidate_n <= 8192:
                try:
                    reference = direct.compute_direct_hbs(
                        shape.curve, shape.derivative, n=2 * candidate_n,
                        pole=pole,
                        negative_mass_tolerance=negative_mass_tolerance,
                        negative_peak_tolerance=negative_peak_tolerance,
                    )
                    reference_n = 2 * candidate_n
                except RuntimeError as exc:
                    # 更细一级连判据都达不到：这一级的「干净」不是收敛，只是
                    # 采样恰好避开了退化处，不能据此交付迹。
                    record["status"] = "rejected_unconverged"
                    record["beta_convergence_note"] = f"2n check failed: {exc}"
                    diagnostics.append(record)
                    previous = (candidate_n, trial)
                    continue
            elif previous is not None:
                reference_n, reference = previous
            elif candidate_n == candidates[-1] and candidate_n > 8:
                # 顶层没有 2n 预算：用 n/2 的宽松解做同一收敛估计。
                try:
                    reference = direct.compute_direct_hbs(
                        shape.curve, shape.derivative, n=candidate_n // 2,
                        pole=pole, negative_mass_tolerance=float("inf"),
                    )
                    reference_n = candidate_n // 2
                except RuntimeError as exc:
                    record["beta_convergence_note"] = f"n/2 check failed: {exc}"
            if reference is not None:
                diff = _beta_gap(candidate_n, trial.beta,
                                 reference_n, reference.beta)
                record["beta_convergence"] = diff
                record["beta_convergence_against_n"] = reference_n
                if diff > angle_tolerance:
                    record["status"] = "rejected_beta"
                    diagnostics.append(record)
                    previous = (candidate_n, trial)
                    continue
        record["status"] = "accepted"
        diagnostics.append(record)
        result, accepted_n = trial, candidate_n
        acceptance = "strict" if strict else "negative_mass"
        beta_convergence = record.get("beta_convergence")
        previous = (candidate_n, trial)
        break
    if result is None:
        reasons = "; ".join(
            f"n={row['n']}: {row.get('error', row.get('status'))}"
            for row in diagnostics
        )
        raise RuntimeError(
            "harmonic-measure density did not reach the acceptance criterion "
            f"({reasons})"
        )

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
        "n": int(accepted_n),
        "n_ladder": list(candidates),
        "density_acceptance": acceptance,
        "min_raw_internal_density": float(result.min_raw_internal_density),
        "min_raw_external_density": float(result.min_raw_external_density),
        "negative_mass_internal": float(result.negative_mass_internal),
        "negative_mass_external": float(result.negative_mass_external),
        "negative_peak_ratio_internal": record["negative_peak_ratio_internal"],
        "negative_peak_ratio_external": record["negative_peak_ratio_external"],
        "beta_convergence": beta_convergence,
        "density_diagnostics": diagnostics,
    }


def harmonic_measure_mu_from_boundary(
    bound: np.ndarray,
    n: int = 1024,
    disk: "DiskMesh | None" = None,
    normalize: str = "default",
) -> tuple[np.ndarray, np.ndarray, "DiskMesh", object, dict]:
    """调和测度路径的便捷入口：`bound → (mu, field, disk, result, stats)`。

    2.2.0 起这是 `hbs._pipeline` 的薄封装，与 `get_hbs(method="harmonic_measure")` 走**同一
    段代码**（此前 `_mu_on_mesh` 是一条独立实现，同一规范化有两份，语义必然漂移）。

    这个入口留给仍走便捷封装的调用方；新代码应直接用 `hbs.get_hbs` 或
    `hbs.get_hbs_result`（后者能拿到引擎名与降级落点）。

    :param `n`: 边界层离散点数（传给 `harmonic_measure_welding`），默认 1024
    :param `disk`: 显式网格；省略时用 `(density=0.01, circle=500)` 的缓存网格
    :param `normalize`: `"default"`（默认）/`"i2"`/`"m34"`/`"c0"`/`"none"`，与 `get_hbs` 同义
    :raises RuntimeError: 全部输入阶梯失败
    """
    from hbs._pipeline import build_hbs, prepare_boundary, resolve_disk
    from hbs._engines import harmonic_measure as _engine

    circle_num = disk.circle_num if disk is not None else 500
    # 与 `get_hbs(method="harmonic_measure")` 一致：调和测度引擎不重采样输入边界。
    prepared = prepare_boundary(np.asarray(bound, float), circle_num, resample=False)
    result = _engine(prepared, circle_num)
    disk, fem = resolve_disk(disk, 0.01, circle_num)
    # `result.values` 已由引擎取样到源圆均匀网格；这里不再重复 `linear_interp`。
    mu, field, welding, disk, gauge = build_hbs(
        result.values, result.welding, disk, fem, normalize
    )
    return (
        mu,
        field,
        disk,
        None,
        {
            "engine": result.engine,
            "rung": result.rung,
            "normalize": normalize,
            "welding": welding,
            "gauge": gauge,
        },
    )
