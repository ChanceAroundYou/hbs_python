"""HBS - Harmonic Beltrami Signature。

由平面形状边界求调和 Beltrami 签名（复系数，定义在三角面心），并从中重建形状。
"""
__version__ = "2.3.2"

from ._pipeline import (
    DEFAULT_NORMALIZE,
    NORMALIZATIONS,
    EngineAgreement,
    HBSResult,
    compare_engines,
)
from .conformal_welding import ConformalWelding, get_conformal_welding
from .gauges import GAUGES, GaugeChoice, select_gauge
from .hbs import get_hbs, get_hbs_result, reconstruct_from_hbs
from .mesh import DiskMesh, Mesh, get_rect, get_unit_disk, get_unit_disk_in_rect
from .qc import get_beltrami_coefficient, lsqc_solver, mu_chop
from .weld_compare import (
    WeldingAgreement,
    WeldingCurve,
    compare_conformal_welding,
    compare_welding,
    welding_curve,
)

from . import conformal_welding, gauges, mesh, qc, utils

__all__ = [
    "__version__",
    # HBS 主流程
    "get_hbs",
    "get_hbs_result",
    "reconstruct_from_hbs",
    "HBSResult",
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
    # 拟共形
    "get_beltrami_coefficient",
    "lsqc_solver",
    "mu_chop",
    # 网格
    "Mesh",
    "DiskMesh",
    "get_rect",
    "get_unit_disk",
    "get_unit_disk_in_rect",
]
