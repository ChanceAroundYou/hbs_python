"""HBS - Harmonic Beltrami Signature。

由平面形状边界求调和 Beltrami 签名（复系数，定义在三角面心），并从中重建形状。
"""
__version__ = "2.6.5"

from ._pipeline import (
    DEFAULT_MESH_PROFILE,
    DEFAULT_NORMALIZE,
    NORMALIZATIONS,
    EngineAgreement,
    HBSResult,
    MeshProfile,
    compare_engines,
)
from .boundary_selection import BoundarySelection, solve_sliding_boundary
from .conformal_welding import ConformalWelding, get_conformal_welding
from .gauges import GAUGES, GaugeChoice, select_gauge
from .hbs import get_hbs, get_hbs_result, reconstruct_from_hbs
from .mesh import DiskMesh, Mesh, get_rect, get_unit_disk, get_unit_disk_in_rect
from .qc import get_beltrami_coefficient, lsqc_solver, mu_chop
from .riemann import RiemannMap, RiemannMapDiagnostics, riemann_map
from .tbs import solve_tbs
from .volumetric import (
    RevolutionChart,
    SpheroidChart,
    TetraMesh,
    ball_mobius,
    ball_tetra_mesh,
    boundary_vertex_k,
    build_tetramesh,
    deformation_data,
    distortion_metrics,
    element_k,
    mobius_scale,
    solve_fixed_trace,
    solve_softmax,
    solve_softmax_continuation,
)
from .weld_compare import (
    WeldingAgreement,
    WeldingCurve,
    compare_conformal_welding,
    compare_welding,
    welding_curve,
)

from . import (
    conformal_welding,
    gauges,
    generalized_hbs,
    mesh,
    qc,
    tbs,
    utils,
    volumetric,
)

__all__ = [
    "__version__",
    # HBS 主流程
    "get_hbs",
    "get_hbs_result",
    "reconstruct_from_hbs",
    "HBSResult",
    # Riemann map (harmonic measure / Green function + Cauchy integral)
    "RiemannMap",
    "RiemannMapDiagnostics",
    "riemann_map",
    # 源圆旋转规范
    "GAUGES",
    "GaugeChoice",
    "select_gauge",
    "DEFAULT_NORMALIZE",
    # 引擎一致性
    "NORMALIZATIONS",
    "EngineAgreement",
    "compare_engines",
    "WeldingAgreement",
    "WeldingCurve",
    "compare_conformal_welding",
    "compare_welding",
    "welding_curve",
    # 共形焊接
    "ConformalWelding",
    "get_conformal_welding",
    # 自由边界对应选择（PDE 路线）
    "BoundarySelection",
    "solve_sliding_boundary",
    # 任意物理基底的广义 HBS（可选 triangle 依赖）
    "generalized_hbs",
    # 拟共形
    "get_beltrami_coefficient",
    "lsqc_solver",
    "mu_chop",
    # 网格
    "Mesh",
    "DiskMesh",
    "MeshProfile",
    "DEFAULT_MESH_PROFILE",
    "get_rect",
    "get_unit_disk",
    "get_unit_disk_in_rect",
    # 固定完整圆周 TBS（局部有限 P1 候选，无证书、无全局或连续结论）
    "solve_tbs",
    # 三维固定迹原型（离散，非连续极值）
    "TetraMesh",
    "build_tetramesh",
    "ball_tetra_mesh",
    "SpheroidChart",
    "RevolutionChart",
    "ball_mobius",
    "mobius_scale",
    "deformation_data",
    "distortion_metrics",
    "element_k",
    "boundary_vertex_k",
    "solve_fixed_trace",
    "solve_softmax",
    "solve_softmax_continuation",
]
