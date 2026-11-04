"""进程内 disk/fem 对象缓存与 FEM 分解复用的回归测试。"""

from __future__ import annotations

import numpy as np

from hbs import cache
from hbs.extension import fem_extend


def test_load_unit_disk_reuses_same_objects():
    disk1, fem1 = cache.load_unit_disk(0.05, 120)
    disk2, fem2 = cache.load_unit_disk(0.05, 120)
    assert disk1 is disk2
    assert fem1 is fem2


def test_fem_extension_operator_is_cached():
    disk, fem = cache.load_unit_disk(0.05, 120)
    first = cache.fem_extension_operator(disk, fem)
    second = cache.fem_extension_operator(disk, fem)
    assert first is second


def test_fem_extend_reuses_cached_factor():
    disk, fem = cache.load_unit_disk(0.05, 120)
    theta = 2.0 * np.pi * np.arange(disk.circle_num) / disk.circle_num
    values = np.exp(1j * theta)
    _field1, factor1 = fem_extend(disk, values, fem=fem)
    _field2, factor2 = fem_extend(disk, values, fem=fem)
    assert factor1 is factor2