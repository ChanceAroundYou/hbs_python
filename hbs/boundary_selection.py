"""自由边界对应选择：用单侧 Fourier 条件确定 ``γ ∘ p`` 的参数化 ``p``。

给定一条正向 Jordan 曲线 ``γ``（及其导数），本模块寻找参数化

    p(θ) = θ + c₀ + Σ_{k=1}^{M} (a_k cos kθ + b_k sin kθ)

使复合曲线 ``γ ∘ p`` 的 Fourier 谱只保留常数项与一次项，且一次项相对于
输入曲线自身的一阶 Fourier 系数为实正——这是**外侧共形可延拓性**的隐式写法：

- 不输入外侧密度，也不输入预先给定的边界对应 ``β``；
- 但它并没有"消除"外侧共形结构，只是把该条件写成了有限个禁用频率的零点；
- 有限频率上的零残差不是连续误差证书。

求解器是带阻尼线搜索的 Gauss–Newton，并约束采样参数速度为正。它只保证**局部**
收敛；固定单一高阶从零初值起步可能在某一阶停在线搜索。默认调用会按
``8 → 16 → 32 → …`` 逐级加密，直到加密残差低于 ``target_residual`` 或到达
``max_modes``；传入 ``initial`` 时则关闭自动续算，由调用方控制 warm start。

本模块只负责边界对应，不包含 HBS 求解、调和延拓或 μ 重建。典型用法::

    import numpy as np
    from hbs.boundary_selection import solve_sliding_boundary

    curve = lambda t: 1.2 * np.exp(1j * t) + 0.1 * np.exp(-2j * t)
    derivative = lambda t: 1.2j * np.exp(1j * t) - 0.2j * np.exp(-2j * t)

    selection = solve_sliding_boundary(curve, derivative)
    print(selection.diagnostics["accepted"],
          selection.diagnostics["dense_forbidden_residual"])

    theta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    boundary = curve(selection.parameter(theta))

``selection.diagnostics["accepted"]`` 只是「加密谱残差与采样单调性阈值通过」，
不是全局最优、唯一性或连续意义的证明。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, lstsq

__all__ = [
    "BoundarySelection",
    "solve_sliding_boundary",
]


def _positive_int(value, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from exc
    if result < 1:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


@dataclass
class BoundarySelection:
    """一次边界选择的结果。

    Parameters
    ----------
    coefficients:
        长度 ``2 * modes + 1`` 的 Fourier 系数，顺序为
        ``[c₀, a₁, …, a_M, b₁, …, b_M]``。
    modes:
        使用的最高频率 ``M``。
    diagnostics:
        停止原因、迭代数、离散/加密残差、采样最小参数速度等诊断量。
    """

    coefficients: np.ndarray
    modes: int
    diagnostics: dict

    def parameter(self, theta):
        """把参考参数 ``θ`` 映到选定的边界参数 ``p(θ)``。"""
        theta = np.asarray(theta, dtype=float)
        basis, _ = _parameter_basis(theta.ravel(), self.modes)
        return (theta.ravel() + basis @ self.coefficients).reshape(theta.shape)


def _parameter_basis(theta, modes):
    """返回参数扰动 ``p(θ) - θ`` 的基函数及其导数。

    基的列顺序为 ``[1, cos θ, …, cos Mθ, sin θ, …, sin Mθ]``。
    """
    modes = _positive_int(modes, "modes")
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional")
    k = np.arange(1, modes + 1)
    phase = theta[:, None] * k
    basis = np.c_[np.ones(len(theta)), np.cos(phase), np.sin(phase)]
    derivative = np.c_[np.zeros(len(theta)), -k * np.sin(phase), k * np.cos(phase)]
    return basis, derivative


class _SlidingBoundaryProblem:
    """固定曲线下的有限 Fourier 边界选择问题。

    目标泛函是禁用正频率的加权平方和；``evaluate`` 同时给出解析 Jacobian，
    供 :func:`solve_sliding_boundary` 的 Gauss–Newton 使用。
    """

    def __init__(self, curve, derivative, modes=32, collocation_factor=8):
        if not callable(curve) or not callable(derivative):
            raise TypeError("curve and derivative must be callable")
        self.curve, self.derivative = curve, derivative
        self.modes = _positive_int(modes, "modes")
        self.collocation_factor = _positive_int(
            collocation_factor, "collocation_factor"
        )
        self.count = self.collocation_factor * (self.modes + 1)
        if self.count <= 2 * self.modes + 1:
            raise ValueError(
                "collocation_factor is too small for the requested modes"
            )
        self.theta = 2 * np.pi * np.arange(self.count) / self.count
        self.basis, self.basis_derivative = _parameter_basis(self.theta, self.modes)
        dense_theta = 2 * np.pi * np.arange(4 * self.count) / (4 * self.count)
        _, self.dense_derivative = _parameter_basis(dense_theta, self.modes)
        self.positive = np.arange(2, 2 * self.modes + 2)
        self.weights = np.sqrt(self.positive)
        z = np.asarray(curve(self.theta), dtype=complex)
        if z.ndim != 1 or len(z) != self.count or not np.all(np.isfinite(z)):
            raise ValueError("curve must return a finite 1-D array on the collocation grid")
        self.scale = float(np.sqrt(np.mean(abs(z - z.mean()) ** 2)))
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("curve is degenerate: its scale must be finite and positive")
        # 一阶系数的形状内参考相位。用它做归一化后，“一阶为实正”变成相对于
        # 输入曲线自身朝向的条件：整体旋转输入时 reference 同步旋转，选出的
        # 边界参数化不变（直接固定到绝对实轴会把源圆旋转绑到坐标系上）。
        self.reference = np.fft.fft(z)[1] / self.count

    def evaluate(self, coefficients, jacobian=False):
        """在给定系数处求残差、约束与可选的 Jacobian。"""
        coefficients = np.asarray(coefficients, dtype=float)
        expected = 2 * self.modes + 1
        if coefficients.shape != (expected,):
            raise ValueError(f"coefficients must have shape ({expected},)")
        p = self.theta + self.basis @ coefficients
        spectrum = np.fft.fft(self.curve(p)) / self.count
        forbidden = spectrum[self.positive] * self.weights / self.scale
        leading = spectrum[1] / self.reference
        residual = np.r_[forbidden.real, forbidden.imag, leading.imag]
        result = {
            "residual": residual,
            "cost": float(residual @ residual),
            "min_speed": float(np.min(1 + self.dense_derivative @ coefficients)),
            "leading": leading,
            "parameter": p,
        }
        if jacobian:
            dspectrum = (
                np.fft.fft(self.derivative(p)[:, None] * self.basis, axis=0) / self.count
            )
            blocked = dspectrum[self.positive] * self.weights[:, None] / self.scale
            result["jacobian"] = np.vstack(
                [
                    blocked.real,
                    blocked.imag,
                    (dspectrum[1] / self.reference).imag[None, :],
                ]
            )
        return result


def _newton_step(jacobian, residual):
    """解最小二乘 ``min ||J δ - r||``，返回 ``(δ, source)``。

    Jacobian 实际是 ``(4M+1) × (2M+1)`` 的高矩阵（禁用频率的实/虚部各算一个
    方程），所以不能直接 `solve`。正常路径用法方程 ``JᵀJ δ = Jᵀr`` 的 Cholesky
    分解；若法方程病态/非正定，或回代残差/步长不可信，回退到 ``gelsy``。
    ``source`` 会写进诊断，便于定位“这次为什么慢”。
    """
    try:
        normal = jacobian.T @ jacobian
        rhs = jacobian.T @ residual
        factor = cho_factor(normal, lower=True, check_finite=False)
        delta = cho_solve(factor, rhs, check_finite=False)
        if np.all(np.isfinite(delta)):
            normal_residual = float(np.linalg.norm(normal @ delta - rhs))
            normal_reference = max(float(np.linalg.norm(rhs)), 1.0)
            residual_norm = float(np.linalg.norm(jacobian @ delta - residual))
            residual_reference = max(float(np.linalg.norm(residual)), 1.0)
            step_norm = float(np.linalg.norm(delta))
            if (
                normal_residual <= 1e-10 * normal_reference
                and residual_norm <= 1.1 * residual_reference
                and step_norm <= 1e6 * residual_reference
            ):
                return delta, "normal"
    except (LinAlgError, ValueError):
        pass

    delta = lstsq(jacobian, residual, cond=1e-12, lapack_driver="gelsy")[0]
    if not np.all(np.isfinite(delta)):
        raise RuntimeError("boundary selection Newton step is non-finite")
    return delta, "lstsq"


def _solve_once(
    curve,
    derivative,
    modes,
    initial,
    max_iterations,
    tolerance,
    collocation_factor=8,
):
    """单阶求解（内部函数；不自动续算）。"""
    max_iterations = _positive_int(max_iterations, "max_iterations")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    start = perf_counter()
    problem = _SlidingBoundaryProblem(
        curve, derivative, modes, collocation_factor=collocation_factor
    )
    coefficients = np.zeros(2 * problem.modes + 1)
    if initial is not None:
        initial = np.asarray(initial, dtype=float)
        if initial.ndim != 1 or len(initial) % 2 != 1:
            raise ValueError("initial must be a 1-D array with an odd length")
        old_modes = (len(initial) - 1) // 2
        keep = min(old_modes, problem.modes)
        coefficients[0] = initial[0]
        coefficients[1 : keep + 1] = initial[1 : keep + 1]
        coefficients[problem.modes + 1 : problem.modes + 1 + keep] = initial[
            old_modes + 1 : old_modes + 1 + keep
        ]
    iterations, reason = 0, "iteration_limit"
    step_sources = []
    for iteration in range(max_iterations):
        iterations = iteration + 1
        state = problem.evaluate(coefficients, True)
        if np.sqrt(state["cost"]) < tolerance:
            reason = "residual_tolerance"
            break
        delta, source = _newton_step(state["jacobian"], -state["residual"])
        step_sources.append(source)
        accepted = False
        for power in range(20):
            step = 2.0**-power
            trial = problem.evaluate(coefficients + step * delta)
            if (
                trial["min_speed"] > 1e-7
                and trial["leading"].real > 0
                and trial["cost"] < state["cost"] * (1 - 1e-4 * step)
            ):
                coefficients += step * delta
                accepted = True
                break
        if not accepted:
            reason = "line_search_stalled"
            break
    # 独立的更密 Fourier 检查；采样单调性不是连续证明。
    theta = 2 * np.pi * np.arange(4 * problem.count) / (4 * problem.count)
    basis, dbasis = _parameter_basis(theta, problem.modes)
    p = theta + basis @ coefficients
    spectrum = np.fft.fft(curve(p)) / len(p)
    pos = np.arange(2, len(p) // 2)
    residual_dense = float(np.sqrt(np.sum(pos * abs(spectrum[pos]) ** 2)) / problem.scale)
    min_speed = float(np.min(1 + dbasis @ coefficients))
    leading = spectrum[1] / problem.reference
    diagnostics = {
        "seconds": perf_counter() - start,
        "modes": problem.modes,
        "collocation_count": problem.count,
        "stop_reason": reason,
        "iterations": iterations,
        "newton_fallbacks": sum(source == "lstsq" for source in step_sources),
        "newton_step_sources": step_sources,
        "discrete_residual": float(np.sqrt(problem.evaluate(coefficients)["cost"])),
        "dense_forbidden_residual": residual_dense,
        "dense_min_parameter_speed": min_speed,
        "leading_real": float(leading.real),
        "leading_imag": float(leading.imag),
        "accepted": bool(residual_dense < 2e-7 and min_speed > 0 and leading.real > 0),
    }
    return BoundarySelection(coefficients, problem.modes, diagnostics)


def _stage_modes(modes, max_modes, min_modes=8):
    """续算频率表：``8, 16, 32, …``，最后一项是请求的上限。

    ``min_modes`` 只控制“最早允许停止”的阶数，不改变逐级加密的起点；低阶
    残差若已低于绝对阈值，仍可在低阶提前停止（见 ``min_absolute_residual``）。
    """
    max_modes = _positive_int(max_modes, "max_modes")
    min_modes = _positive_int(min_modes, "min_modes")
    if min_modes > max_modes:
        raise ValueError(
            f"min_modes ({min_modes}) must not exceed max_modes ({max_modes})"
        )
    if modes is None:
        limit = max_modes
    else:
        limit = _positive_int(modes, "modes")
        if limit > max_modes:
            raise ValueError(f"modes ({limit}) must not exceed max_modes ({max_modes})")
    stages = []
    current = 8
    while current < limit:
        stages.append(current)
        current *= 2
    stages.append(limit)
    return stages


def solve_sliding_boundary(
    curve,
    derivative,
    modes=None,
    initial=None,
    max_iterations=50,
    tolerance=2e-10,
    *,
    continuation=True,
    max_modes=512,
    min_modes=8,
    min_absolute_residual=0.0,
    target_residual=2e-7,
    collocation_factor=8,
):
    """用阻尼 Gauss–Newton 选择边界参数化。

    Parameters
    ----------
    curve, derivative:
        以参数为自变量的复值曲线及其导数，均需可向量化。
    modes:
        最高 Fourier 频率 ``M``；``None``（默认）表示自动续算到 ``accepted``
        或 ``target_residual``，上限 ``max_modes``。
    initial:
        可选的初值系数（长度 ``2 * M_old + 1``）。**提供 initial 时不做自动续算**，
        只在该阶跑一次，保留调用方对 warm start 的完全控制。
    max_iterations:
        每一阶的最大 Gauss–Newton 迭代数。
    tolerance:
        单阶离散残差范数的停止阈值。
    continuation:
        为 ``True`` 且 ``initial is None`` 时，按 ``8 → 16 → 32 → …`` 逐级加密；
        ``False`` 时只跑 ``modes`` 这一阶（要求 ``modes`` 是整数）。
    max_modes:
        自动续算的最高频率上限，默认 512。
    min_modes:
        自动续算的最低频率下限，默认 8。PDE 引擎用 64：低阶残差过目标时，
        高阶项仍可能影响焊接/μ，不能在第 8 阶就停。
    min_absolute_residual:
        若某阶的加密残差已经低于此绝对阈值，即使还没到 ``min_modes`` 也允许
        停止；默认 0（关闭该捷径）。PDE 引擎用 ``2e-8``，让圆/近似解析形状
        在低阶直接停，而普通形状仍至少走到 ``min_modes``。
    target_residual:
        自动续算的加密残差目标，默认 ``2e-7``（与 ``accepted`` 同一阈值；
        可调紧到 ``2e-9`` 换取更高精度，代价是更高阶与更长时间）。
    collocation_factor:
        每个 Fourier 系数对应的配点数倍数，默认 8；必须大于
        ``(2 * modes + 1) / (modes + 1)``。4 是实验性下采样，需自行验证
        dense 残差/焊接精度。

    Returns
    -------
    BoundarySelection
        ``coefficients`` 为选定参数化的 Fourier 系数，``diagnostics`` 给出
        停止原因、迭代数、离散与加密频率残差、采样最小参数速度等。
        使用自动续算时，``diagnostics["continuation_stages"]`` 记录每一阶的
        落点，``diagnostics["seconds"]`` 是整条续算的总时间。

    Notes
    -----
    这是**局部**求解器：线搜索失败时以 ``stop_reason="line_search_stalled"``
    返回当前点。固定单一高阶从零初值起步容易停在线搜索；默认的逐级续算就是
    为避免这一点。不要只看 ``accepted``，还要检查
    ``dense_forbidden_residual`` 与 ``dense_min_parameter_speed``。
    """
    max_iterations = _positive_int(max_iterations, "max_iterations")
    target_residual = float(target_residual)
    if not np.isfinite(target_residual) or target_residual <= 0.0:
        raise ValueError("target_residual must be finite and positive")
    min_absolute_residual = float(min_absolute_residual)
    if not np.isfinite(min_absolute_residual) or min_absolute_residual < 0.0:
        raise ValueError("min_absolute_residual must be finite and non-negative")
    if initial is not None:
        initial = np.asarray(initial, dtype=float)
        if initial.ndim != 1 or len(initial) % 2 != 1:
            raise ValueError("initial must be a 1-D array with an odd length")
        if modes is None:
            modes = max(1, (len(initial) - 1) // 2)
    if not continuation or initial is not None:
        if modes is None:
            raise ValueError("modes must be an integer when continuation is disabled")
        return _solve_once(
            curve, derivative, modes, initial, max_iterations, tolerance,
            collocation_factor=collocation_factor,
        )

    start = perf_counter()
    stages = _stage_modes(modes, max_modes, min_modes)
    current = None
    history = []
    selection = None
    for stage in stages:
        selection = _solve_once(
            curve, derivative, stage, current, max_iterations, tolerance,
            collocation_factor=collocation_factor,
        )
        current = selection.coefficients
        diagnostics = selection.diagnostics
        history.append(
            {
                "modes": stage,
                "stop_reason": diagnostics["stop_reason"],
                "iterations": diagnostics["iterations"],
                "accepted": diagnostics["accepted"],
                "dense_forbidden_residual": diagnostics["dense_forbidden_residual"],
                "seconds": diagnostics["seconds"],
            }
        )
        if (
            modes is None
            and diagnostics["dense_forbidden_residual"] < target_residual
            and (
                stage >= min_modes
                or diagnostics["dense_forbidden_residual"] < min_absolute_residual
            )
        ):
            break
    selection.diagnostics["seconds"] = perf_counter() - start
    selection.diagnostics["continuation_stages"] = history
    return selection
