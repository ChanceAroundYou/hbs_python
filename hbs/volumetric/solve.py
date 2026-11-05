"""三维固定边界迹的离散极小极大求解器。

两个求解器解的是同一个离散问题：边界迹固定，内部顶点的三个坐标全部自由，

    minimize    max_e K_e,      K_e = max(K_O,e, K_I,e)

:func:`solve_fixed_trace` 用 SLSQP 直接解图形式（epigraph 变量 ``t ≥ log K_e``），
约束 Jacobian 是稠密的 ``(3m, 3n+1)``；:func:`solve_softmax` 把 max 换成
log-sum-exp、用 L-BFGS-B 解光滑上界，内存只需一列梯度。两者都只保证是
**局部**解：本模块不声称连续性、唯一性或全局最优。

``solve_softmax`` 的默认光滑形式是 ``smoothing="pair"``，即对 ``2m`` 个 entry
一起做 log-sum-exp::

    S_β(x) = (1/β) log Σ_e [ exp(β log K_O,e) + exp(β log K_I,e) ]

它对 ``x`` 处处可微，并且 ``log M ≤ S_β ≤ log M + log(2m)/β``。旧的
``smoothing="max"`` 只对 ``m`` 个 ``max(K_O,K_I)`` 求和，界是 ``log(m)/β`` 更紧，
但 ``max`` 在 ``K_O=K_I`` 处不可微；实测三维极小点恰好落在该折面上，解析梯度
只有单侧导数，``|∇S|`` 永远到不了 0。需要复现 2026-11-04 之前读数时用
``smoothing="max"``。
"""
from __future__ import annotations

from collections import deque
from time import perf_counter

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .mesh import TetraMesh, as_tetramesh
from .metrics import deformation_data, distortion_metrics

SMOOTHINGS = ("pair", "max")


class _Stall(Exception):
    """内部信号：callback 里跳出 L-BFGS-B（其返回值不保证被采纳）。"""


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
    smoothing: str = "pair",
    ftol: float = 0.0,
    gtol: float = 1e-12,
    maxcor: int = 100,
    maxls: int = 60,
    maxfun: int | None = None,
    stall_window: int = 0,
    stall_rel: float = 1e-12,
) -> dict:
    """log-sum-exp 光滑化的 ``min max K``（L-BFGS-B，带盒约束）。

    目标 ``(1/β) log Σ exp(β log K)``，``β → ∞`` 时逼近 ``ess sup``；有限 ``β``
    给出的是它的光滑上界，不是原极小极大问题的精确解。

    ``smoothing="pair"``（默认）对 ``2m`` 个 ``K_O``/``K_I`` entry 一起求和：

        ``S_β = (1/β) log Σ_e [exp(β log K_O,e) + exp(β log K_I,e)]``

    处处可微，``log M ≤ S_β ≤ log M + log(2m)/β``。``smoothing="max"`` 是旧行为
    （``log M ≤ S_β ≤ log M + log(m)/β``）：界更紧，但 ``max`` 在 ``K_O=K_I``
    处不可微，而实测极小点正落在该折面上，解析梯度只有单侧导数、``|∇S|`` 不会
    趋于 0。

    停止条件：默认 ``ftol=0``（关掉 L-BFGS-B 的**单步**相对下降判据——目标值
    ``F≈1`` 时它等价于绝对阈值 ``~1e-16``，在仍是 1e-4/步 的时候就会宣布收敛），
    ``gtol=1e-12``，并显式传 ``maxfun``（缺省 ``20*maxiter``）——scipy 默认
    ``maxfun=15000`` 会在加大 ``maxiter`` 时静默截断。``maxcor``/``maxls`` 加大
    以容纳病态的 ``β`` 目标。需要「值不再变化就停」时设 ``stall_window>0``：
    窗口内相对改善不超过 ``stall_rel`` 即停止（对近似平坦的高 ``β`` 目标有用）。

    注意 ``ftol=0`` 时仍可能收到 scipy 的
    ``CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH``：此时
    ``factr = ftol/eps = 0``，触发条件是相对下降 ``≤ 0``，即**舍入误差水平上的
    停滞**，不是某个 1e-10 阈值。``stopped_by`` 按 scipy 的消息原文归类，读日志
    时要按这条区分。
    """
    mesh = as_tetramesh(mesh)
    if mesh.trace is None:
        raise ValueError("mesh.trace is required for solve_softmax")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if smoothing not in SMOOTHINGS:
        raise ValueError(f"smoothing must be one of {SMOOTHINGS}, got {smoothing!r}")
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")
    if stall_window < 0:
        raise ValueError("stall_window must be non-negative")
    if stall_rel < 0:
        raise ValueError("stall_rel must be non-negative")

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
    m = len(tetra)
    values = initial.copy()
    counter = {"calls": 0}

    def objective(x):
        counter["calls"] += 1
        values[:n] = x.reshape(n, 3)
        _, U, s, Vh, _, KO, KI = deformation_data(values, mesh)
        if smoothing == "pair":
            logK = np.log(np.maximum(np.concatenate([KO, KI]), 1e-15))
        else:
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
        if smoothing == "pair":
            dphi = dO * weight[:m, None, None] + dI * weight[m:, None, None]
        else:
            dphi = np.where((KO >= KI)[:, None, None], dO, dI) * weight[:, None, None]

        gradient = np.zeros((n, 3))
        for local in range(4):
            node = tetra[:, local]
            valid = node < n
            vec = np.einsum("tij,tj->ti", dphi[valid], gradients[valid, local])
            np.add.at(gradient, node[valid], vec)
        return value, gradient.ravel()

    if maxfun is None:
        maxfun = 20 * maxiter
    x0 = initial[:n].ravel()
    bounds = [(float(v - delta), float(v + delta)) for v in x0]
    window = deque(maxlen=stall_window) if stall_window > 0 else None
    state = {"hit": False, "iterations": 0, "x": None, "fun": None}

    def callback(intermediate_result):
        # scipy 1.18 的 L-BFGS-B 不理会 callback 的 True 返回值（实测仍跑到
        # maxiter），所以停窗口用异常跳出；最后一次迭代点由 state 保存。
        state["iterations"] += 1
        state["x"] = np.asarray(intermediate_result.x, dtype=float).copy()
        state["fun"] = float(intermediate_result.fun)
        if window is None:
            return
        window.append(state["fun"])
        if len(window) == stall_window:
            drop = window[0] - window[-1]
            if drop <= stall_rel * max(1.0, abs(window[-1])):
                state["hit"] = True
                raise _Stall

    started = perf_counter()
    try:
        result = minimize(
            objective,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options=dict(
                maxiter=maxiter,
                maxfun=maxfun,
                ftol=ftol,
                gtol=gtol,
                maxcor=maxcor,
                maxls=maxls,
            ),
        )
        x_final = np.asarray(result.x, dtype=float)
        fun = float(result.fun)
        nit = int(result.nit)
        nfev = int(getattr(result, "nfev", counter["calls"]))
        message = str(result.message)
        success = bool(result.success)
        jac = np.asarray(getattr(result, "jac", np.zeros(1)), dtype=float)
    except _Stall:
        x_final = state["x"]
        fun = state["fun"]
        _, grad = objective(x_final)
        nit = int(state["iterations"])
        nfev = int(counter["calls"])
        message = (
            f"stall: relative decrease over {stall_window} iterations "
            f"<= {stall_rel}"
        )
        success = False
        jac = np.asarray(grad, dtype=float)
        result = OptimizeResult(
            x=x_final,
            fun=fun,
            jac=jac,
            nit=nit,
            nfev=nfev,
            status=99,
            success=False,
            message=message,
        )
    values[:n] = x_final.reshape(n, 3)
    values[n:] = trace
    if state["hit"]:
        stopped_by = "stall"
    elif "ITERATIONS" in message:
        stopped_by = "maxiter"
    elif "EVALUATIONS" in message:
        stopped_by = "maxfun"
    elif "GRADIENT" in message:
        stopped_by = "gtol"
    elif "REDUCTION" in message or "CONVERGENCE" in message:
        stopped_by = "ftol"
    else:
        stopped_by = "other"
    return dict(
        result=result,
        values=values,
        seconds=perf_counter() - started,
        evaluations=counter["calls"],
        metrics=distortion_metrics(values, mesh),
        nit=nit,
        nfev=nfev,
        message=message,
        success=success,
        objective_value=fun,
        beta=float(beta),
        smoothing=smoothing,
        gradient_inf=float(np.abs(jac).max()) if jac.size else float("nan"),
        stopped_by=stopped_by,
        maxiter=int(maxiter),
        maxfun=int(maxfun),
        ftol=float(ftol),
        gtol=float(gtol),
        stall_window=int(stall_window),
    )


def solve_softmax_continuation(
    mesh: TetraMesh | dict,
    initial: np.ndarray,
    betas=(10, 20, 40, 80),
    maxiter: int = 400,
    delta: float = 0.02,
    callback=None,
    **solver_options,
) -> dict:
    """按 ``β`` 递增依次求解，把上一阶段的结果当地下一阶段的初值。

    这是求解策略而不是新的数学问题：每个 ``β`` 都只给光滑上界，最终读数仍取
    最后一个阶段的 ``distortion_metrics``。``solver_options`` 原样转给
    :func:`solve_softmax`（``smoothing`` / ``ftol`` / ``gtol`` / ``maxfun`` /
    ``maxcor`` / ``maxls`` / ``stall_window`` / ``stall_rel``）。
    """
    betas = tuple(float(beta) for beta in betas)
    if not betas:
        raise ValueError("betas must not be empty")
    values = np.asarray(initial, dtype=float).copy()
    history = []
    for beta in betas:
        stage = solve_softmax(
            mesh, values, beta=beta, maxiter=maxiter, delta=delta, **solver_options
        )
        values = stage["values"]
        history.append(
            dict(
                beta=beta,
                nit=stage["nit"],
                nfev=stage["nfev"],
                message=stage["message"],
                stopped_by=stage["stopped_by"],
                gradient_inf=stage["gradient_inf"],
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
