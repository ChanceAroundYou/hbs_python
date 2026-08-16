"""Quasiconformal（拟共形）求解子包：Beltrami 系数与 LSQC 求解器。"""
from .beltrami import get_beltrami_coefficient, mu_chop
from .lsqc import lsqc_solver

# 向后兼容：旧路径 hbs.qc.bc 仍可用（shim 模块）
from . import bc  # noqa: F401

__all__ = ["get_beltrami_coefficient", "lsqc_solver", "mu_chop"]
