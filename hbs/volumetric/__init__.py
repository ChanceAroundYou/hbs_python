"""三维固定边界迹的拟共形原型（离散）。

这是 ``HBS-research/研究/三维TBS.md`` 的**候选**签名在离散网格上的实现，不是二维
HBS/TBS 的已证明推广，也不是连续意义下的极值求解器：

- 源域是单位球内的 ``TetraMesh``；目标边界先取一张二维共形球面图作为迹
  （:class:`SpheroidChart` / :class:`RevolutionChart`）；
- 体内用 P1 分片仿射映射，在固定边界迹的所有映照里最小化
  ``max_e K_e``，``K = max(s1³/|J|, |J|/s3³)``；
- 两个求解器（:func:`solve_fixed_trace` / :func:`solve_softmax`）都只给
  **局部**解；不承诺唯一性、连续性或全局最优。

``K`` 是纯几何量：``K = 1`` 当且仅当该单元是相似变换。它不含材料参数，
因此不能当作应力或应变读数。

示例::

    from hbs.volumetric import SpheroidChart, ball_tetra_mesh, solve_fixed_trace

    chart = SpheroidChart(a=1.0, c=1.5)
    mesh = ball_tetra_mesh(boundary_count=48, interior_count=10, seed=0)
    result = solve_fixed_trace(chart, mesh, maxiter=40)
    result["final"]["max_K"]
"""
from __future__ import annotations

from .audit import audit_discrete_gradient, audit_polyhedral_boundary
from .chart import RevolutionChart, SpheroidChart, ball_mobius, mobius_scale
from .mesh import (
    TetraMesh,
    as_tetramesh,
    ball_tetra_mesh,
    build_tetramesh,
    element_gradients,
)
from .metrics import (
    boundary_vertex_k,
    deformation_data,
    distortion_metrics,
    element_k,
)
from .solve import solve_fixed_trace, solve_softmax, solve_softmax_continuation

__all__ = [
    # 网格
    "TetraMesh",
    "as_tetramesh",
    "ball_tetra_mesh",
    "build_tetramesh",
    "element_gradients",
    # 边界图
    "SpheroidChart",
    "RevolutionChart",
    "ball_mobius",
    "mobius_scale",
    # 畸变
    "deformation_data",
    "distortion_metrics",
    "element_k",
    "boundary_vertex_k",
    # 求解
    "solve_fixed_trace",
    "solve_softmax",
    "solve_softmax_continuation",
    # 诊断
    "audit_discrete_gradient",
    "audit_polyhedral_boundary",
]
