r"""任意物理基底 \(B\) 上的广义 HBS：边界 welding + B 上的 Dirichlet 延拓。

定义（本模块实现的对象）
------------------------
给定有界单连通 Jordan 域 \(B,T\)，取两域各自的内、外共形参数化
\(\Phi_X^{\rm in},\Phi_X^{\rm out}\)，在**边界迹**上定义
\[
f_1:B\to T,\quad f_2:B^c\to T^c,\qquad
F=(f_1|_{\partial B})^{-1}\circ(f_2|_{\partial B})
:\partial B\to\partial B .
\]
随后在物理域 \(B\) 上解逐分量 Dirichlet 问题
\[
\Delta H=0\ \text{in }B,\qquad H|_{\partial B}=F,
\]
并取 \(\mu=H_{\bar z}/H_z\)。

边界迹的实现口径
----------------
默认引擎 `engine="harmonic_measure"` 用
`hbs._harmonic_measure.direct.compute_direct_hbs`，在一条光滑 Jordan 边界上
同时给出内、外调和测度相位 \(\alpha_X(t),\beta_X(t)\)，其中 \(t\) 为归一化
弧长参数。它们正是 \(\arg\Phi_X^{\rm in}(\gamma_X(t))\) 与
\(\arg\Phi_X^{\rm out}(\gamma_X(t))\)。于是

* \(f_2(\gamma_B(t))\) 对应 \(T\) 边界上满足 \(\beta_T(s)=\beta_B(t)\) 的点；
* 该点在 \(T\) 的内相位为 \(\alpha_T(s)\)；
* \(f_1^{-1}\) 把它送回满足 \(\alpha_B(t')=\alpha_T(s)\) 的 \(\gamma_B(t')\)。

所以 welding 只需四次单调相位插值，不组装焊接表、不在圆盘上解 PDE：

\[
F(\gamma_B(t))
=\gamma_B\!\left(
\alpha_B^{-1}\!\left(\alpha_T\!\left(\beta_T^{-1}(\beta_B(t))\right)\right)
\right).
\]

相位引擎（`engine=`）
--------------------
可选 `engine="pde"`：用 `hbs.pde.pde_welding` 的内域平坦度量 Robin 问题
+ 外域 Fourier 选择器给出同一对 \(\alpha,\beta\)。它不要求域是圆盘：
`pde_welding` 在目标边界上解 trace PDE。返回的 `parameter`/`init_y` 通过
周期展开与 `_pde_phase_chart` 变成与调和测度同格式的相位网格；后续的
边界 welding 与 \(B\) 上的 Dirichlet 延拓完全复用。两个引擎只差源圆旋转，
12 个形状上焊接最大差 \(4\times10^{-4}\)–\(4\times10^{-2}\)，μ 统计一致到
\(10^{-3}\) 量级；叶片 `bc` 的尖端会让 PDE 内域求解失败。

鲁棒性阶梯（`engine="auto"`, `rung="auto"`）
-------------------------------------------
`rung` 依次尝试 `raw` → `resample500` → `+s2` → `+s4` → `+s8` → `+s16`；
`engine="auto"` 先走调和测度、再走 PDE。第一个成功的 `(engine, rung)` 被
记录在图上，调用方必须把 `rung != "raw"` 视为**几何已平滑**，不能把结果
当作原始边界的高保真解。该阶梯把此前失败的 `real_fish-3`、`real_apple-1`
救回到 HM 的平滑档；叶片仍由 HM raw 处理。

规范化
------
库的 `compute_direct_hbs` 已对**全部配点**做 M1 Möbius 居中
（\(\sum_k \Phi_1^{-1}(z_k)=0\)），且配点参数 \(t\) 是归一化弧长，因此这里
的 M1 就是“物理弧长测度下的全局重心规范”，不需要额外基点或标记点。
居中只固定 Möbius 平移自由度，仍余一个整体旋转；本模块按当前阶段约定
**保留该旋转等价类**，不施加最终单值 gauge。外相位 \(\beta\) 由无穷远极点
的对数单层方程确定，对应外映射的无穷远规范。

证据标签：实现为 [D] 定义 + [X] 有限离散实验；仅对光滑 Jordan 边界声明。
物理 \(B\) 的三角化需要可选依赖 `triangle`；核心 HBS 不依赖它。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from hbs._harmonic_measure import direct
from hbs.cache import _fem_operator, fem_extension_operator
from hbs.extension import fem_extend
from hbs.harmonic_measure import _SplineShape
from hbs.hbs import _resample_boundary
from hbs.mesh import DiskMesh

TAU = 2.0 * np.pi

#: 相位引擎的输入阶梯；与库 `_engines._rungs` 的平滑档对齐。
RUNG_NAMES = (
    "raw",
    "resample500",
    "resample500+s2",
    "resample500+s4",
    "resample500+s8",
    "resample500+s16",
)
ENGINE_NAMES = ("harmonic_measure", "pde")


def _apply_rung(bound: np.ndarray, rung: str) -> np.ndarray:
    """在已定向/锚定的边界上应用输入阶梯；返回 (m,2) 浮点数组。"""
    if rung not in RUNG_NAMES:
        raise ValueError(f"unknown rung {rung!r}; expected one of {RUNG_NAMES}")
    if rung == "raw":
        return np.asarray(bound, float)
    resampled = _resample_boundary(np.asarray(bound, float), 500)
    if rung == "resample500":
        return resampled
    sigma = float(rung.rsplit("s", 1)[1])
    return np.column_stack([
        gaussian_filter1d(resampled[:, 0], sigma, mode="wrap"),
        gaussian_filter1d(resampled[:, 1], sigma, mode="wrap"),
    ])


def _periodic_interp(x, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """在周期单调网格上做线性插值；要求 xp/fp 从 0 递增到 2π。"""
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    if xp.shape != fp.shape or xp.ndim != 1 or len(xp) < 2:
        raise ValueError("periodic grids must be equal-length 1-D arrays")
    if not (np.all(np.diff(xp) > 0.0) and np.all(np.diff(fp) > 0.0)):
        raise ValueError("periodic grids must be strictly increasing")
    if not (np.isclose(xp[0], 0.0) and np.isclose(fp[0], 0.0)):
        raise ValueError("periodic grids must start at zero")
    if not (xp[-1] < TAU and fp[-1] < TAU):
        raise ValueError("periodic grids must end below 2*pi")
    xp_ext = np.r_[xp, xp[0] + TAU]
    fp_ext = np.r_[fp, fp[0] + TAU]
    return np.interp(np.mod(np.asarray(x, float), TAU), xp_ext, fp_ext)


def _canonical_roll(raw: np.ndarray) -> np.ndarray:
    """把闭合边界滚到几何确定的起点：离边界质心最远的点。

    库的 α、β 都锚在拟合参数 t=0，也就是输入数组的第一个样本。若直接使用
    用户给的顺序，同一几何在不同起点下会得到不同的相位代表，F 随之改变。
    先滚到与输入顺序无关的物理点，α、β 的锚点就固定在同一几何位置。

    “离质心最远”在平移、正缩放、旋转下都等变，因此不像“最低 y”那样依赖
    坐标轴。高度对称形状（椭圆、圆）可能有多个并列最远点，此时用坐标
    lexsort 作确定性 tie-break；这类形状上的旋转会改变所选分支，是已知的
    规范限制，不是数值误差。
    """
    center = raw.mean(axis=0)
    radius = np.hypot(raw[:, 0] - center[0], raw[:, 1] - center[1])
    peak = radius.max()
    tied = np.flatnonzero(radius >= peak - 1e-12 * max(peak, 1e-300))
    order = np.lexsort((raw[tied, 0], raw[tied, 1]))
    return np.roll(raw, -int(tied[order[0]]), axis=0)


@dataclass
class DomainChart:
    r"""单个 Jordan 域的内、外边界相位图（不构造圆盘 PDE）。

    `engine="harmonic_measure"` 用库的边界积分相位（默认）；`engine="pde"`
    用 `hbs.pde.pde_welding` 的内域平坦度量 Robin PDE + 外域 Fourier 选择器。
    两个引擎给出同一对相位 \(\alpha(t),\beta(t)\)，只差源圆旋转规范。
    """

    boundary: np.ndarray
    shape: _SplineShape
    result: direct.DirectHBS | None
    t: np.ndarray
    points: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    engine: str = "harmonic_measure"
    rung: str = "raw"
    pde_diagnostics: dict | None = None

    def eval_boundary(self, t: np.ndarray) -> np.ndarray:
        """按本图的参数化取物理边界点。

        PDE 引擎的 `points` 与 `t` 是 PDE 选出的非均匀网格，因此用周期线性
        插值；调和测度引擎仍用原来的样条。
        """
        t = np.mod(np.asarray(t, float), TAU)
        if self.engine == "pde":
            xp = np.r_[self.t, self.t[0] + TAU]
            fp = np.r_[self.points, self.points[0]]
            return (np.interp(t, xp, fp.real)
                    + 1j * np.interp(t, xp, fp.imag))
        return np.asarray(self.shape.curve(t), complex)

    @classmethod
    def from_boundary(
        cls,
        bound: np.ndarray,
        n: int = 512,
        *,
        engine: str = "harmonic_measure",
        rung: str = "raw",
        pde_n: int | None = None,
    ) -> "DomainChart":
        """构造相位图；`engine="auto"` / `rung="auto"` 走失败阶梯。

        尝试顺序：引擎按 `harmonic_measure` → `pde`，每个引擎内按
        `raw` → `resample500` → `+s2` → `+s4` → `+s8` → `+s16`。
        第一个成功的结果被返回，并把实际 `engine` / `rung` 记在图上。
        """
        n = int(n)
        if n < 64 or n % 2:
            raise ValueError("n must be an even integer >= 64")
        if engine not in ENGINE_NAMES + ("auto",):
            raise ValueError(
                f"engine must be one of {ENGINE_NAMES + ('auto',)}, got {engine!r}"
            )
        if rung not in RUNG_NAMES + ("auto",):
            raise ValueError(
                f"rung must be one of {RUNG_NAMES + ('auto',)}, got {rung!r}"
            )
        raw = np.asarray(bound, float)
        if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 8:
            raise ValueError("bound must be an n-by-2 smooth closed curve sample")
        if not np.all(np.isfinite(raw)):
            raise ValueError("bound must contain only finite coordinates")
        raw = _canonical_roll(raw)
        z = raw[:, 0] + 1j * raw[:, 1]
        edges = np.abs(np.diff(np.r_[z, z[0]]))
        area = abs(0.5 * np.sum(z.real * np.roll(z.imag, -1)
                                - z.imag * np.roll(z.real, -1)))
        if edges.min() <= 1e-14 or area <= 1e-14:
            raise ValueError("degenerate boundary: zero area or repeated points")

        engines = ENGINE_NAMES if engine == "auto" else (engine,)
        rungs = RUNG_NAMES if rung == "auto" else (rung,)
        errors: list[str] = []
        for engine_name in engines:
            for rung_name in rungs:
                try:
                    processed = _apply_rung(raw, rung_name)
                    return cls._build(
                        processed, n=n, engine=engine_name,
                        pde_n=pde_n, rung=rung_name,
                    )
                except Exception as exc:  # 记录失败，继续下一档
                    errors.append(
                        f"{engine_name}/{rung_name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        raise RuntimeError(
            "no engine/rung succeeded:\n  " + "\n  ".join(errors)
        )

    @classmethod
    def _build(
        cls,
        processed: np.ndarray,
        *,
        n: int,
        engine: str,
        pde_n: int | None,
        rung: str,
    ) -> "DomainChart":
        shape = _SplineShape(processed)
        if engine == "pde":
            t, points, alpha, beta, diagnostics = _pde_phase_chart(
                shape, n=n, pde_n=pde_n
            )
            return cls(
                boundary=processed,
                shape=shape,
                result=None,
                t=t,
                points=points,
                alpha=alpha,
                beta=beta,
                engine=engine,
                rung=rung,
                pde_diagnostics=diagnostics,
            )
        result = direct.compute_direct_hbs(
            shape.curve,
            shape.derivative,
            n=n,
            pole=direct._interior_pole(shape.points),
        )
        t = TAU * np.arange(n) / n
        return cls(
            boundary=processed,
            shape=shape,
            result=result,
            t=t,
            points=np.asarray(result.boundary, complex),
            alpha=np.asarray(result.alpha, float),
            beta=np.asarray(result.beta, float),
            engine=engine,
            rung=rung,
        )


def _pde_phase_chart(
    shape: _SplineShape, n: int, pde_n: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """从 `pde_welding` 的源角/边界参数/焊接值提取 (t, points, alpha, beta)。"""
    from hbs.pde import pde_welding

    bound = np.column_stack([shape.points.real, shape.points.imag])
    welding = pde_welding(bound, circle_num=n, n=int(pde_n or n))
    theta = np.asarray(welding["init_x_angle"], float)
    t = np.unwrap(np.asarray(welding["parameter"], float))
    y = np.asarray(welding["init_y"], complex)
    alpha = np.unwrap(np.angle(y))
    points = np.asarray(welding["boundary"], complex)
    if not (len(theta) == len(t) == len(alpha) == len(points)):
        raise RuntimeError("pde welding returned inconsistent array lengths")
    if not (np.all(np.diff(t) > 0.0)
            and np.all(np.diff(theta) > 0.0)
            and np.all(np.diff(alpha) > 0.0)):
        raise RuntimeError("pde phase grids are not strictly monotone")
    t = t - t[0]
    alpha = alpha - alpha[0]
    keep = len(t)
    while keep > 3 and (t[keep - 1] >= TAU or alpha[keep - 1] >= TAU
                        or theta[keep - 1] >= TAU):
        keep -= 1
    return (
        t[:keep].copy(),
        points[:keep].copy(),
        alpha[:keep].copy(),
        theta[:keep].copy(),
        dict(welding.get("pde_diagnostics", {})),
    )


@dataclass
class BoundaryWelding:
    """`F=(f1|∂B)^{-1}∘(f2|∂B)` 在基底边界配点上的取值。"""

    base: DomainChart
    target: DomainChart
    t: np.ndarray
    target_parameter: np.ndarray
    values: np.ndarray
    base_points: np.ndarray


def boundary_welding(
    base_bound: np.ndarray,
    target_bound: np.ndarray,
    n: int = 512,
    *,
    engine: str = "harmonic_measure",
    rung: str = "raw",
    pde_n: int | None = None,
) -> BoundaryWelding:
    """构造物理边界自映射 F；相位全为严格单调，因此 F 保向。"""
    base = DomainChart.from_boundary(
        base_bound, n=n, engine=engine, rung=rung, pde_n=pde_n
    )
    target = DomainChart.from_boundary(
        target_bound, n=n, engine=engine, rung=rung, pde_n=pde_n
    )

    s = _periodic_interp(base.beta, target.beta, target.t)
    inner_value = _periodic_interp(s, target.t, target.alpha)
    t_in = _periodic_interp(inner_value, base.alpha, base.t)

    increments = np.diff(np.r_[t_in, t_in[0] + TAU])
    if np.min(increments) <= 0.0:
        raise RuntimeError("welding lost strict monotonicity; refine n")

    values = base.eval_boundary(t_in)
    return BoundaryWelding(
        base=base,
        target=target,
        t=base.t.copy(),
        target_parameter=t_in,
        values=values,
        base_points=base.points.copy(),
    )


def _dist_to_poly(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a = poly[None, :]
    b = np.r_[poly[1:], poly[:1]][None, :]
    ab = b - a
    t = np.clip(
        ((points[:, None] - a) * np.conj(ab)).real
        / np.maximum(np.abs(ab) ** 2, 1e-300),
        0.0,
        1.0,
    )
    return np.min(np.abs(points[:, None] - (a + t * ab)), axis=1)


def _points_in_polygon(points: np.ndarray, poly: np.ndarray,
                       chunk: int = 4096) -> np.ndarray:
    """射线法判断点是否在多边形内；不引入 matplotlib 依赖。"""
    points = np.asarray(points, complex).ravel()
    poly = np.asarray(poly, complex).ravel()
    if len(poly) < 3:
        return np.zeros(len(points), bool)
    x1, y1 = poly.real, poly.imag
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    inside = np.zeros(len(points), bool)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        x = p.real[:, None]
        y = p.imag[:, None]
        crosses = (y1[None, :] > y) != (y2[None, :] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = ((x2 - x1)[None, :] * (y - y1[None, :])
                   / (y2 - y1)[None, :] + x1[None, :])
        inside[start:start + chunk] = (
            np.count_nonzero(crosses & (x < xin), axis=1) % 2
        ).astype(bool)
    return inside


def _seed_points(boundary: np.ndarray, h: float) -> np.ndarray:
    xmin, xmax = boundary.real.min(), boundary.real.max()
    ymin, ymax = boundary.imag.min(), boundary.imag.max()
    xs = np.arange(xmin + h, xmax, h)
    ys = np.arange(ymin + h, ymax, h)
    grid_x, grid_y = np.meshgrid(xs, ys)
    candidates = (grid_x + 1j * grid_y).ravel()
    candidates = candidates[_points_in_polygon(candidates, boundary)]
    keep = candidates[_dist_to_poly(candidates, boundary) > 0.6 * h]
    return np.column_stack([keep.real, keep.imag])


def _build_mesh(boundary: np.ndarray, h: float) -> DiskMesh:
    n = len(boundary)
    if h <= 0.0:
        raise ValueError("h must be positive")
    try:
        import triangle as tr
    except ImportError as exc:
        raise ImportError(
            "generalized HBS needs the optional 'triangle' package; "
            "install it with `pip install hbs[generalized]`"
        ) from exc
    seeds = _seed_points(boundary, h)
    vertices = np.vstack([np.column_stack([boundary.real, boundary.imag]), seeds])
    segments = np.column_stack([np.arange(n), (np.arange(n) + 1) % n])
    out = tr.triangulate({"vertices": vertices, "segments": segments}, "pq30YQ")
    verts = np.asarray(out["vertices"], float)
    faces = np.asarray(out["triangles"], int)
    if not np.allclose(verts[:n], vertices[:n], atol=1e-12):
        raise RuntimeError("triangle did not preserve the boundary vertex order")
    return DiskMesh(circle=verts[:n], in_vert=verts[n:], face=faces)


@dataclass
class GeneralizedHBSResult:
    welding: BoundaryWelding
    mesh: DiskMesh
    field: np.ndarray
    mu_raw: np.ndarray
    jacobian: np.ndarray
    boundary_order_preserved: bool
    nonpositive_jacobian: int
    nonfinite_mu: int


def solve_generalized_hbs(
    base_bound: np.ndarray,
    target_bound: np.ndarray,
    n: int = 512,
    h: float = 0.02,
    *,
    engine: str = "harmonic_measure",
    rung: str = "raw",
    pde_n: int | None = None,
) -> GeneralizedHBSResult:
    r"""边界 welding → 物理 B 网格上的 Dirichlet 延拓 → 逐面 μ。

    μ 用库的同一离散公式 `Dc f / Dz f`，但**不做** `mu_chop` 饱和：
    诊断必须看到真实的 \(|\mu|\ge1\)，不能把截断值当作合法映射。
    """
    welding = boundary_welding(
        base_bound, target_bound, n=n, engine=engine, rung=rung, pde_n=pde_n
    )
    mesh = _build_mesh(welding.base_points, h=h)
    fem = _fem_operator(mesh)
    factor = fem_extension_operator(mesh, fem)
    field, _ = fem_extend(mesh, welding.values, fem=fem, factor=factor)
    f = field[:, 0] + 1j * field[:, 1]

    f_z = mesh.Dz @ f
    f_zbar = mesh.Dc @ f
    good = np.abs(f_z) > 1e-14
    mu_raw = np.full(f_z.shape, np.nan + 1j * np.nan, complex)
    mu_raw[good] = f_zbar[good] / f_z[good]
    jacobian = np.abs(f_z) ** 2 - np.abs(f_zbar) ** 2

    increments = np.diff(np.r_[welding.target_parameter,
                               welding.target_parameter[0] + TAU])
    return GeneralizedHBSResult(
        welding=welding,
        mesh=mesh,
        field=f,
        mu_raw=mu_raw,
        jacobian=jacobian,
        boundary_order_preserved=bool(np.min(increments) > 0.0),
        nonpositive_jacobian=int(np.count_nonzero(jacobian <= 0.0)),
        nonfinite_mu=int(np.count_nonzero(~np.isfinite(mu_raw))),
    )