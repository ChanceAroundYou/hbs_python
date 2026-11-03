"""三维固定边界迹的离散极小极大求解器。

两个求解器解的是同一个离散问题：边界迹固定，内部顶点的三个坐标全部自由，

    minimize    max_e K_e,      K_e = max(K_O,e, K_I,e)

:func:`solve_fixed_trace` 用 SLSQP 直接解图形式（epigraph 变量 ``t ≥ log K_e``），
约束 Jacobian 是稠密的 ``(3m, 3n+1)``；:func:`solve_softmax` 把 max 换成
log-sum-exp、用 L-BFGS-B 解光滑上界，内存只需一列梯度。两者都只保证是
**局部**解：本模块不声称连续性、唯一性或全局最优。
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .mesh import TetraMesh, as_tetramesh
from .metrics import deformation_data, distortion_metrics


def solve_fixed_trace(
    chart,
    mesh: TetraMesh | dict,
    maxiter: int = 60,
    ftol: float = 1e-8,
    initial_values: np.ndarray | None = None,
    callback=None,
    bounds_delta: float | None = None,
) -> dict:
    """离散极小极大：固定边界迹，自由内部顶点。

    参数
    ----
    chart
        提供 ``boundary(unit)`` 与 ``cone_initial(points)`` 的边界图
        （:class:`~hbs.volumetric.SpheroidChart` / ``RevolutionChart``）。
    mesh
        ``TetraMesh`` 或历史字典；内部顶点必须排在边界顶点前面。
    bounds_delta
        为 ``None`` 时内部坐标不设界（研究仓原型的行为）。给一个正值时，
        每个坐标被限制在初值 ``±bounds_delta`` 内——稠密约束的 SLSQP 在
        约束数远大于变量数时容易第一步就飞出线性化范围，加盒约束才可控。
        这是收敛策略，不是数学约束；改变它会改变迭代路径与读数。

    返回字典含 ``success`` / ``message`` / ``initial`` / ``final`` / ``values`` /
    ``epigraph_K`` / ``boundary_residual`` / ``iterations`` / ``evaluations``。

    注意：返回的 ``values`` 是 SLSQP 的**最后一个迭代点**，不保证可行。
    迭代上限或线性搜索失败时，它可能比 ``initial`` 还差，且
    ``minimum_constraint < 0`` 表示 epigraph 约束被违反。要拿最佳可行点，
    请在 ``callback`` 里自行保存（研究仓 `run_blade_tbs.py` 就是这么做的）。
    ``final`` 只是该点的读数，不代表收敛值。
    """
    mesh = as_tetramesh(mesh)
    started = perf_counter()
    n = mesh.interior_count
    tetra = mesh.tetra
    m = len(tetra)
    if n < 1:
        raise ValueError("mesh has no interior vertices to optimize")
    if m < 1:
        raise ValueError("mesh has no tetrahedra")

    trace = np.asarray(chart.boundary(mesh.points[n:]), dtype=float)
    if trace.shape != (len(mesh.points) - n, 3):
        raise ValueError("chart.boundary returned the wrong shape")

    if initial_values is None:
        values = np.asarray(chart.cone_initial(mesh.points), dtype=float)
    else:
        values = np.asarray(initial_values, dtype=float).copy()
        if values.shape != mesh.points.shape or not np.all(np.isfinite(values)):
            raise ValueError(
                "initial_values must be finite and have shape (source_vertices, 3)"
            )
        if not np.allclose(values[n:], trace, rtol=1e-12, atol=1e-12):
            raise ValueError("initial_values must keep the prescribed boundary trace")
        if np.min(deformation_data(values, mesh)[4]) <= 0:
            raise ValueError(
                "initial_values must have positive Jacobian on every tetrahedron"
            )
    values[n:] = trace
    initial = values.copy()
    initial_metrics = distortion_metrics(initial, mesh)
    if initial_metrics["min_J"] <= 0:
        return dict(
            success=False,
            message="analytic cone P1 initialization has flipped tets",
            initial=initial_metrics,
            final=initial_metrics,
            seconds=perf_counter() - started,
            iterations=0,
            boundary_residual=0.0,
            initial_values=initial,
            values=values,
        )

    jfloor = min(1e-7, initial_metrics["min_J"] * 1e-4)
    x0 = np.r_[values[:n].ravel(), np.log(initial_metrics["max_K"]) + 1e-7]
    last_x = None
    cached = None
    calls = 0

    def evaluate(x):
        nonlocal last_x, cached, calls
        if last_x is not None and np.array_equal(x, last_x):
            return cached
        calls += 1
        values[:n] = x[:-1].reshape(n, 3)
        _, U, s, Vh, J, KO, KI = deformation_data(values, mesh)
        invT = np.einsum("tik,tk,tkj->tij", U, 1 / s, Vh)
        dO = (
            3 * np.einsum("ti,tj->tij", U[:, :, 0], Vh[:, 0, :]) / s[:, 0, None, None]
            - invT
        )
        dI = (
            invT
            - 3
            * np.einsum("ti,tj->tij", U[:, :, -1], Vh[:, -1, :])
            / s[:, -1, None, None]
        )
        dJ = J[:, None, None] * invT
        constraint = np.r_[x[-1] - np.log(KO), x[-1] - np.log(KI), J - jfloor]
        jac = np.zeros((3 * m, 3 * n + 1))
        jac[: 2 * m, -1] = 1.0
        for local in range(4):
            node = tetra[:, local]
            valid = node < n
            rows = np.flatnonzero(valid)
            grad = mesh.gradients[valid, local]
            for offset, derivative in ((0, -dO), (m, -dI), (2 * m, dJ)):
                vec = np.einsum("tij,tj->ti", derivative[valid], grad)
                jac[(rows + offset)[:, None], 3 * node[valid, None] + np.arange(3)] = vec
        cached = (constraint, jac)
        last_x = x.copy()
        return cached

    objective_jac = np.zeros_like(x0)
    objective_jac[-1] = 1.0
    if bounds_delta is None:
        bounds = [(None, None)] * (3 * n) + [(0.0, None)]
    else:
        delta = float(bounds_delta)
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("bounds_delta must be a positive finite number")
        bounds = [(float(v - delta), float(v + delta)) for v in x0[:-1]] + [(0.0, None)]

    opt = minimize(
        lambda x: x[-1],
        x0,
        jac=lambda x: objective_jac,
        method="SLSQP",
        constraints=[
            dict(type="ineq", fun=lambda x: evaluate(x)[0], jac=lambda x: evaluate(x)[1])
        ],
        bounds=bounds,
        callback=callback,
        options=dict(maxiter=maxiter, ftol=ftol, disp=False),
    )
    values[:n] = opt.x[:-1].reshape(n, 3)
    final_metrics = distortion_metrics(values, mesh)
    constraint = evaluate(opt.x)[0]
    return dict(
        success=bool(opt.success),
        message=str(opt.message),
        initial=initial_metrics,
        final=final_metrics,
        seconds=perf_counter() - started,
        iterations=int(opt.nit),
        evaluations=calls,
        epigraph_K=float(np.exp(opt.x[-1])),
        minimum_constraint=float(np.min(constraint)),
        boundary_residual=float(np.max(np.abs(values[n:] - initial[n:]))),
        initial_values=initial,
        values=values.copy(),
    )


def solve_softmax(
    mesh: TetraMesh | dict,
    initial: np.ndarray,
    beta: float = 60.0,
    maxiter: int = 400,
    delta: float = 0.02,
) -> dict:
    """log-sum-exp 光滑化的 ``min max K``（L-BFGS-B，带盒约束）。

    目标：``min_x (1/β) log Σ_e exp(β log K_e)``。``β → ∞`` 时逼近
    ``ess sup``；有限 ``β`` 给出的是它的光滑上界，不是原极小极大问题的精确解。
    梯度与 SLSQP 版本用同一个解析式，但一次只累加一列，不再构造
    ``(3m, 3n+1)`` 的稠密 Jacobian。
    """
    mesh = as_tetramesh(mesh)
    if mesh.trace is None:
        raise ValueError("mesh.trace is required for solve_softmax")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")

    n = mesh.interior_count
    initial = np.asarray(initial, dtype=float)
    if initial.shape != mesh.points.shape:
        raise ValueError(
            f"initial must have shape {mesh.points.shape}, got {initial.shape}"
        )
    trace = np.asarray(mesh.trace, dtype=float)
    if trace.shape != (len(mesh.points) - n, 3):
        raise ValueError("mesh.trace must have shape (boundary_vertices, 3)")
    if not np.allclose(initial[n:], trace, rtol=1e-12, atol=1e-12):
        raise ValueError("initial must keep the prescribed boundary trace")

    tetra, gradients = mesh.tetra, mesh.gradients
    values = initial.copy()
    counter = {"calls": 0}

    def objective(x):
        counter["calls"] += 1
        values[:n] = x.reshape(n, 3)
        _, U, s, Vh, _, KO, KI = deformation_data(values, mesh)
        logK = np.log(np.maximum(np.maximum(KO, KI), 1e-15))
        z = beta * logK
        top = z.max()
        weight = np.exp(z - top)
        total = weight.sum()
        value = top / beta + np.log(total) / beta
        weight /= total

        invT = np.einsum("tik,tk,tkj->tij", U, 1 / s, Vh)
        dO = (
            3 * np.einsum("ti,tj->tij", U[:, :, 0], Vh[:, 0, :]) / s[:, 0, None, None]
            - invT
        )
        dI = (
            invT
            - 3
            * np.einsum("ti,tj->tij", U[:, :, -1], Vh[:, -1, :])
            / s[:, -1, None, None]
        )
        dphi = np.where((KO >= KI)[:, None, None], dO, dI) * weight[:, None, None]

        gradient = np.zeros((n, 3))
        for local in range(4):
            node = tetra[:, local]
            valid = node < n
            vec = np.einsum("tij,tj->ti", dphi[valid], gradients[valid, local])
            np.add.at(gradient, node[valid], vec)
        return value, gradient.ravel()

    x0 = initial[:n].ravel()
    bounds = [(float(v - delta), float(v + delta)) for v in x0]
    started = perf_counter()
    result = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-8),
    )
    values[:n] = result.x.reshape(n, 3)
    values[n:] = trace
    return dict(
        result=result,
        values=values,
        seconds=perf_counter() - started,
        evaluations=counter["calls"],
        metrics=distortion_metrics(values, mesh),
        nit=int(result.nit),
        message=str(result.message),
        success=bool(result.success),
    )


def solve_softmax_continuation(
    mesh: TetraMesh | dict,
    initial: np.ndarray,
    betas=(10, 20, 40, 80),
    maxiter: int = 400,
    delta: float = 0.02,
    callback=None,
) -> dict:
    """按 ``β`` 递增依次求解，把上一阶段的结果当地下一阶段的初值。

    这是求解策略而不是新的数学问题：每个 ``β`` 都只给光滑上界，最终读数仍取
    最后一个阶段的 ``distortion_metrics``。
    """
    betas = tuple(float(beta) for beta in betas)
    if not betas:
        raise ValueError("betas must not be empty")
    values = np.asarray(initial, dtype=float).copy()
    history = []
    for beta in betas:
        stage = solve_softmax(mesh, values, beta=beta, maxiter=maxiter, delta=delta)
        values = stage["values"]
        history.append(
            dict(
                beta=beta,
                nit=stage["nit"],
                message=stage["message"],
                seconds=stage["seconds"],
                **stage["metrics"],
            )
        )
        if callback is not None:
            callback(stage)
    return dict(
        values=values,
        metrics=distortion_metrics(values, mesh),
        history=history,
        seconds=float(sum(item["seconds"] for item in history)),
    )
