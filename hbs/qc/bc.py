"""向后兼容 shim：Beltrami 系数计算已迁至 hbs.qc.beltrami。

保持 `from hbs.qc.bc import get_beltrami_coefficient` 等旧导入可用。
新代码请从 hbs.qc.beltrami 导入。
"""
from .beltrami import get_beltrami_coefficient, mu_chop

__all__ = ["get_beltrami_coefficient", "mu_chop"]
