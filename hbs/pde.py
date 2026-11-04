"""内域平坦度量 PDE + 外域 Fourier 条件：`method="pde"` 引擎。

本模块实现研究仓 ``boundary_flat_metric.solve_boundary_metric_trace`` 的
trace-only 路径：在边界上解非线性 Robin 问题

    Δu = 0 在 Ω 内，    κ + ∂_n u = e^u 在 ∂Ω 上，

其中 ``κ`` 是边界曲率、``n`` 是外法向。解出的 ``u`` 给出内域平坦度量的
边界对数密度；积分一次得到内域共形角 ``α(t)``。外域参数化由
:func:`hbs.boundary_selection.solve_sliding_boundary` 给出，两者组合成
不经过调和测度密度的独立焊接。

## 适用范围与限制

- 只支持光滑 Jordan 边界；尖点、噪声轮廓会让曲率/Robin 残差超限，调用方应
  回退到输入阶梯或换引擎。
- 这是离散原型：验收看 ``curvature_robin_max``、``normalization_residual``、
  ``augmentation_norm`` 三个诊断；引擎用 ``robin/augmentation <= 1e-3`` 的
  宽松阈值，不声称连续误差界或全局唯一性。
- 外域选择器在引擎里用 ``max_modes=192``（硬形状的加密残差可能到 1e-5，
  但 μ 与 512 阶一致）；稠密 `n x n` 线性求解在部分 BLAS 上很慢，默认 n=96。
- 旋转协变性已修好：外域一阶相位用**形状内参考**归一化，整体旋转输入时选出
  的边界参数化逐位不变，μ 相对差约 1e-11。

## M1 对齐与引擎契约

内域 PDE 求解器自带的 sample-moment 条件只覆盖输入样本子集，不能替代论文
M1（``Σ Φ₁⁻¹(z_k)=0``）。``pde_welding`` 在全部 ``n`` 个配点上再做一次
M1 圆盘自同构居中，之后与 :mod:`hbs.harmonic_measure` 在源圆旋转意义下一致：

- bent_lobe / asym_dent3 / ellipse 的焊接差 8e-4° / 0.35° / 8e-6°；
- μ rms 0.0009 / 0.018 / 8e-7，均低于 ``_engines`` 的 1°/0.1 门槛。

## 已知域外输入

强非凸边界可能让内域 Robin 方程停在局部解：研究仓的 ``_oval``
（``cos t + 0.3 cos 2t + i(sin t + 0.2 sin 3t)``）在 n=96 时 Robin 残差约
0.13，引擎显式失败，不静默降级。需要这类形状时请换 ``method`` 或先平滑。
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root

from hbs._harmonic_measure.direct import _operator

__all__ = ["pde_welding", "solve_metric_trace"]


class _BoundaryShape:
    """给 PDE 求解器用的最小 shape 适配器。

    复用 :class:`hbs.harmonic_measure._SplineShape` 的周期样条，并补上
    ``sample_t``（输入采样点的弧长参数），供归一化矩使用。
    """

    def __init__(self, bound: np.ndarray):
        from hbs.harmonic_measure import _SplineShape

        self._spline = _SplineShape(bound)
        self.points = self._spline.points
        edges = np.abs(np.diff(np.r_[self.points, self.points[0]]))
        s = np.r_[0.0, np.cumsum(edges)]
        if s[-1] <= 0:
            raise ValueError("boundary has zero length")
        self.sample_t = 2.0 * np.pi * s / s[-1]

    def curve(self, t):
        return self._spline.curve(t)

    def derivative(self, t):
        return self._spline.derivative(t)


def _integration_matrix(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(t, dt, integration)``；``integration @ f`` 是 f 的谱原函数。"""
    dt = 2.0 * np.pi / n
    t = dt * np.arange(n)
    k = np.fft.fftfreq(n, 1.0 / n)
    ih = np.zeros(n, complex)
    mask = (k != 0) & (np.abs(k) < n / 2)
    ih[mask] = 1.0 / (1j * k[mask])
    integration = np.fft.ifft(
        ih[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0
    ).real
    integration -= integration[0]
    integration += t[:, None] / n
    return t, dt, integration


def solve_metric_trace(
    shape,
    n: int = 512,
    initial=None,
    max_iterations: int = 35,
    tolerance: float = 1e-10,
    *,
    robin_tolerance: float = 1e-6,
    moment_tolerance: float = 1e-8,
    augmentation_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """解内域平坦度量 PDE，返回 ``(u, alpha, diagnostics)``。

    ``u`` 是边界对数密度，``alpha`` 是内域共形角（绝对角，``alpha(0)=0``）。
    ``diagnostics["geometry_pde_accepted"]`` 要求三个原始残差同时过阈值：
    Robin 残差 ``< robin_tolerance``（默认 1e-6）、归一化矩
    ``< moment_tolerance``（默认 1e-8）、增广乘子范数
    ``< augmentation_tolerance``（默认 1e-6）。``pde_welding`` 用 1e-3 的
    宽松阈值，因为 n=96 时硬形状的 Robin 残差约 1e-4，而 μ 与 n=512 一致。
    """
    if n % 2:
        raise ValueError("n must be even")
    if n < 64:
        raise ValueError("n must be at least 64")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    start = perf_counter()
    t, dt, integration = _integration_matrix(n)
    g = np.asarray(shape.curve(t), complex)
    gp = np.asarray(shape.derivative(t), complex)
    if g.shape != (n,) or gp.shape != (n,):
        raise ValueError("shape.curve/derivative must return one value per grid point")
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(gp))):
        raise ValueError("shape.curve/derivative returned non-finite values")
    speed = np.abs(gp)
    if np.any(speed <= 0):
        raise ValueError("boundary parameterization has zero speed (cusp?)")

    normal = -1j * gp / speed
    k = np.fft.fftfreq(n, 1.0 / n)
    gpp = np.fft.ifft(1j * k * np.fft.fft(gp))
    curvature = np.imag(np.conj(gp) * gpp) / speed**3

    operator = _operator(g, gp)
    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n] = operator
    augmented[:n, n] = 1.0
    augmented[n, :n] = dt
    augmented_lu = lu_factor(augmented)
    inverse = lu_solve(augmented_lu, np.vstack([np.eye(n), np.zeros(n)]))[:n]

    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.real(normal[:, None] / (g[:, None] - g[None, :]))
    np.fill_diagonal(kernel, curvature / 2.0)
    dtn = dt * kernel - np.diag(np.pi / speed)
    dtn_matrix = dtn @ inverse

    sample_idx = np.minimum((shape.sample_t / dt).astype(int), n - 1)
    fraction = shape.sample_t / dt - sample_idx
    integ_sample = (1.0 - fraction[:, None]) * integration[sample_idx] + fraction[
        :, None
    ] * np.vstack([integration, np.full((1, n), dt)])[sample_idx + 1]

    u = (
        np.full(n, np.log(2.0 * np.pi / (dt * speed.sum())))
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )
    if u.shape != (n,) or not np.all(np.isfinite(u)):
        raise ValueError("initial must be a finite array with one value per grid point")
    multipliers = np.zeros(2)
    trace = []

    def evaluate(values, lam, jacobian=False):
        ee = np.exp(np.clip(values, -700.0, 100.0))
        alpha = integration @ (ee * speed)
        sample_alpha = integ_sample @ (ee * speed)
        phase = np.exp(1j * sample_alpha)
        moment = phase.mean()
        modes = np.c_[np.cos(alpha), np.sin(alpha)]
        raw = (curvature + dtn_matrix @ values - ee) * speed
        residual = np.r_[raw + modes @ lam, 10.0 * moment.real, 10.0 * moment.imag]
        result = {
            "residual": residual,
            "raw": raw,
            "moment": moment,
            "alpha": alpha,
            "phase": phase,
            "ee": ee,
            "merit": float(np.linalg.norm(residual) / np.sqrt(n)),
        }
        if jacobian:
            d_alpha = integration * (ee * speed)[None, :]
            d_moment = np.mean(
                1j * phase[:, None] * (integ_sample * (ee * speed)[None, :]), axis=0
            )
            matrix = np.zeros((n + 2, n + 2))
            matrix[:n, :n] = (dtn_matrix - np.diag(ee)) * speed[:, None] + (
                -lam[0] * np.sin(alpha) + lam[1] * np.cos(alpha)
            )[:, None] * d_alpha
            matrix[:n, n:] = modes
            matrix[n, :n] = 10.0 * d_moment.real
            matrix[n + 1, :n] = 10.0 * d_moment.imag
            result["matrix"] = matrix
        return result

    for iteration in range(max_iterations):
        state = evaluate(u, multipliers, True)
        trace.append(
            {
                "iteration": iteration,
                "merit": state["merit"],
                "raw_max": float(np.abs(state["raw"]).max()),
            }
        )
        if state["merit"] < tolerance:
            break
        delta = np.linalg.solve(state["matrix"], -state["residual"])
        accepted = False
        for power in range(12):
            step = 2.0**-power
            trial_u = u + step * delta[:n]
            trial_lam = multipliers + step * delta[n:]
            trial = evaluate(trial_u, trial_lam)
            if np.isfinite(trial["merit"]) and trial["merit"] < state["merit"] * (
                1.0 - 1e-4 * step
            ):
                u, multipliers = trial_u, trial_lam
                accepted = True
                break
        if not accepted:
            break

    final = evaluate(u, multipliers)
    geometry_ok = (
        np.abs(final["raw"]).max() < robin_tolerance
        and abs(final["moment"]) < moment_tolerance
        and np.linalg.norm(multipliers) < augmentation_tolerance
    )
    alpha = final["alpha"]
    diagnostics = {
        "seconds": perf_counter() - start,
        "boundary_count": n,
        "curvature_robin_max": float(np.abs(final["raw"]).max()),
        "curvature_robin_rms": float(np.sqrt(np.mean(final["raw"] ** 2))),
        "normalization_residual": float(abs(final["moment"])),
        "augmentation_norm": float(np.linalg.norm(multipliers)),
        "metric_log_min": float(u.min()),
        "metric_log_max": float(u.max()),
        "iterations": len(trace),
        "geometry_pde_accepted": bool(geometry_ok),
        "metric_total_boundary_length": float(dt * np.sum(np.exp(u) * speed)),
        "trace": trace,
    }
    return u, alpha, diagnostics


def _m1_center(alpha: np.ndarray) -> tuple[np.ndarray, complex, float]:
    """论文式 5.11 / Algorithm 6.2 的 M1 规范化：``Σ Φ₁⁻¹(z_k)=0``。

    在全部 ``n`` 个配点上找圆盘自同构 ``a``，使
    ``mean((z-a)/(1-conj(a)z))=0``，再把它作用回内域角。与
    :mod:`hbs._harmonic_measure.direct` 的实现逐式一致；PDE 求解器自带的
    sample-moment 条件只覆盖输入样本子集，不能替代这里的全网格 M1。
    """
    z = np.exp(1j * np.asarray(alpha, complex))
    if z.ndim != 1 or len(z) < 3 or not np.all(np.isfinite(z)):
        raise ValueError("alpha must be a finite 1-D array with at least three points")

    def equations(v):
        a = complex(v[0], v[1])
        mapped = (z - a) / (1 - np.conj(a) * z)
        mean = mapped.mean()
        return np.array([mean.real, mean.imag])

    result = root(equations, np.array([z.mean().real, z.mean().imag]))
    a = complex(result.x[0], result.x[1])
    residual = float(np.linalg.norm(equations(result.x)))
    if not result.success or abs(a) >= 1.0 or residual > 1e-9:
        raise RuntimeError(
            "PDE inner Möbius centering failed "
            f"(|a|={abs(a):.3e}, residual={residual:.3e})"
        )
    centered = np.asarray(alpha, float) + 2.0 * np.angle(
        1.0 - a * np.exp(-1j * np.asarray(alpha, float))
    )
    centered -= centered[0]
    if np.min(np.diff(centered)) <= 0:
        raise RuntimeError("PDE M1-centered angle is not monotone")
    return centered, a, residual


def pde_welding(
    bound: np.ndarray,
    circle_num: int = 500,
    *,
    n: int = 96,
    max_iterations: int = 35,
) -> dict:
    """用「外域 Fourier 条件 + 内域平坦度量 PDE」构造焊接种。

    返回 :meth:`hbs.conformal_welding.ConformalWelding.from_harmonic_measure`
    接受的字典。内域 ``alpha`` 来自 :func:`solve_metric_trace`，外域参数化
    来自 :func:`hbs.boundary_selection.solve_sliding_boundary`；两者组合为

        y(θ) = exp(i (p(θ) + α(p(θ)) - p(θ))) = exp(i α(p(θ))).

    其中 ``p(θ)`` 是外域条件选出的边界参数，``α`` 是内域绝对共形角。
    """
    from hbs.boundary_selection import solve_sliding_boundary

    bound = np.asarray(bound, float)
    if bound.ndim != 2 or bound.shape[1] != 2 or len(bound) < 3:
        raise ValueError("bound must be an n x 2 array with at least three points")
    circle_num = int(circle_num)
    if circle_num < 3:
        raise ValueError("circle_num must be at least 3")

    shape = _BoundaryShape(bound)
    _u, alpha, diagnostics = solve_metric_trace(
        shape,
        n=n,
        max_iterations=max_iterations,
        robin_tolerance=1e-3,
        augmentation_tolerance=1e-3,
    )
    if not diagnostics["geometry_pde_accepted"]:
        raise RuntimeError(
            "PDE interior solve did not meet its raw acceptance thresholds "
            f"(robin={diagnostics['curvature_robin_max']:.2e}, "
            f"moment={diagnostics['normalization_residual']:.2e}, "
            f"augmentation={diagnostics['augmentation_norm']:.2e})"
        )

    alpha, m1_center, m1_residual = _m1_center(alpha)
    diagnostics["m1_center"] = m1_center
    diagnostics["m1_residual"] = m1_residual

    selection = solve_sliding_boundary(
        shape.curve, shape.derivative, max_modes=192
    )
    selection_diagnostics = selection.diagnostics
    if (
        selection_diagnostics["dense_forbidden_residual"] > 1e-4
        or selection_diagnostics["dense_min_parameter_speed"] <= 0
    ):
        raise RuntimeError(
            "boundary selection did not meet its lenient thresholds "
            f"(dense={selection_diagnostics['dense_forbidden_residual']:.2e}, "
            f"speed={selection_diagnostics['dense_min_parameter_speed']:.2e})"
        )

    theta = 2.0 * np.pi * np.arange(circle_num) / circle_num
    parameter = np.mod(selection.parameter(theta), 2.0 * np.pi)
    t, _, _ = _integration_matrix(n)
    relative = CubicSpline(
        np.r_[t, 2.0 * np.pi], np.r_[alpha - t, 0.0], bc_type="periodic"
    )(parameter)
    y = np.exp(1j * (parameter + relative))
    if not np.all(np.isfinite(y)):
        raise RuntimeError("PDE welding produced non-finite boundary values")
    if np.any(np.diff(np.unwrap(np.angle(y))) <= 0):
        raise RuntimeError("PDE welding is not a strictly monotone circle map")

    welding = {
        "init_x_angle": theta,
        "init_x": np.exp(1j * theta),
        "init_y": y,
        "parameter": parameter,
        "boundary": np.asarray(shape.curve(parameter), complex),
    }
    welding["pde_diagnostics"] = diagnostics
    welding["selection_diagnostics"] = selection.diagnostics
    return welding
