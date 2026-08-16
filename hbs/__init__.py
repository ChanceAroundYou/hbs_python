"""HBS - Harmonic Beltrami Signature package.

A Python library for complex analysis and conformal mapping computations.
"""
from .hbs import get_hbs, reconstruct_from_hbs
from .qc import get_beltrami_coefficient, lsqc_solver, mu_chop
from .conformal_welding import ConformalWelding, get_conformal_welding
from .mesh import DiskMesh, Mesh, get_rect, get_unit_disk, get_unit_disk_in_rect

from . import conformal_welding
from . import mesh
from . import qc
from . import utils

__all__ = [
    # HBS 主流程
    "get_hbs",
    "reconstruct_from_hbs",
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
    # 共形焊接
    "ConformalWelding",
    "get_conformal_welding",
]
