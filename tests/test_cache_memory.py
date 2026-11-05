"""进程内 disk/fem 对象缓存与 FEM 分解复用的回归测试。"""

from __future__ import annotations

import numpy as np

import hbs
from hbs import cache
from hbs._pipeline import DEFAULT_CIRCLE, DEFAULT_DENSITY
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


def test_clear_memory_cache_drops_objects_and_factors():
    disk1, fem1 = cache.load_unit_disk(0.05, 120)
    factor1 = cache.fem_extension_operator(disk1, fem1)
    cache.clear_memory_cache()

    assert cache.memory_cache_info()["disk_entries"] == 0
    assert cache.memory_cache_info()["factor_entries"] == 0

    disk2, fem2 = cache.load_unit_disk(0.05, 120)
    factor2 = cache.fem_extension_operator(disk2, fem2)
    assert disk2 is not disk1
    assert factor2 is not factor1
    cache.clear_memory_cache()


def test_disk_memory_is_bounded():
    cache.clear_memory_cache()
    for offset in range(cache._MAX_DISK_ENTRIES + 2):
        cache.load_unit_disk(0.2, 32 + offset)
    info = cache.memory_cache_info()
    assert info["disk_entries"] == info["disk_limit"]
    cache.clear_memory_cache()


def test_factor_memory_is_bounded():
    cache.clear_memory_cache()
    for offset in range(cache._MAX_FACTOR_ENTRIES + 2):
        disk, fem = cache.load_unit_disk(0.2, 32 + offset)
        cache.fem_extension_operator(disk, fem)
    info = cache.memory_cache_info()
    assert info["factor_entries"] == info["factor_limit"]
    cache.clear_memory_cache()


def test_default_mesh_profile_matches_public_defaults():
    assert hbs.DEFAULT_MESH_PROFILE.density == DEFAULT_DENSITY
    assert hbs.DEFAULT_MESH_PROFILE.circle_point_num == DEFAULT_CIRCLE
    assert hbs.DEFAULT_MESH_PROFILE.key == (DEFAULT_DENSITY, DEFAULT_CIRCLE)
    assert isinstance(hbs.DEFAULT_MESH_PROFILE, hbs.MeshProfile)
