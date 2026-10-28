"""两条前向路径共用的 **FEM 调和延拓**：单位圆上的边界值 → 盘内顶点值。

## 数学

给定源圆上的边界值 `y(θ)`（单位圆周 `θ` 均匀采样），求盘内的调和函数 `H`：

    ΔH = 0  在盘内，   H|∂D = y

P1 有限元离散后就是稀疏线性系统 `A h = -B y`，其中 `A` 是内部节点的弱拉普拉斯
（质量加权），`B` 是内部—边界耦合块。`A` 只依赖网格，**与形状无关**，所以
`hbs.cache` 把它跟网格一起存盘、跨进程复用。

## 为什么统一到 FEM

2.1.0 的 zipper 用 `utils.poisson.integral`（稠密 Poisson 核求积），angular 用
P1 FEM。两者都是调和延拓，但：

- 耗时差 **54–63×**（密网格 60796 面：473 ms vs 7.6 ms），因为前者是
  `O(n_inner × n_circle)` 的稠密矩阵乘，后者是稀疏三角求解；
- 延拓值只差 `1.7e-4 ~ 6.3e-4`，但 μ 差 `2.0e-2 ~ 4.9e-2` —— μ = H_z̄/H_z 是
  导数之比，值的误差在导数上放大两个量级。

统一后两条引擎共用同一份 `A`/`B`：焊接之后的延拓、μ、I₂ 规范**只有一份实现**
（`hbs._pipeline`）。两个引擎的差别只在焊接本身（`hbs._engines`），而两条焊接
只相差源圆旋转（`hbs.weld_compare` 度量）。

## 返回约定

`field` 是 `disk.vert_num × 2` 的实数组（`[边界; 内点]`，与 `DiskMesh` 顶点同序）。
zipper 原先返回的 Poisson 结果就是这个形状（`[边界; 内点]`，无外点），所以
`reconstruct_from_hbs` 与下游代码不用改。
"""

from __future__ import annotations

import numpy as np

import hbs.cache as cache


def fem_extend(disk, boundary_values, fem=None, factor=None):
    """把源圆边界值调和延拓到 `disk` 的全部顶点。

    :param `disk`: `DiskMesh`（顶点顺序 `[circle; in_vert; out_vert]`）
    :param `boundary_values`: 长度 `disk.circle_num` 的复数组，圆上一圈的值
    :param `fem`: 可选的 `(A, B)`，来自 `cache.load_unit_disk`；省略则就地计算
    :param `factor`: 可选的 `A` 的 LU 分解；多次求解时传入以免重复分解
    :return: `(field, factor)`，`field` 是 `vert_num × 2` 实数组

    返回值第二项是本次用的 LU 分解，调用方若还要解别的边界值可以直接复用。
    内部保证 `factor` 与 `(A, B)` 配套：省略 `fem` 时会就地构造。
    """
    y = np.asarray(boundary_values, complex).ravel()
    nc = disk.circle_num
    if y.shape[0] != nc:
        raise ValueError(
            f"boundary_values must have {nc} entries (disk.circle_num), got {y.shape[0]}"
        )

    if fem is None:
        fem = cache._fem_operator(disk)
    A, B = fem
    if factor is None:
        factor = cache.fem_extension_operator(disk, fem)

    rhs = -(B @ y)
    inner = factor.solve(rhs.real) + 1j * factor.solve(rhs.imag)

    values = np.empty(disk.vert_num, complex)
    values[:nc] = y
    values[nc:] = inner[: disk.in_vert_num]
    return np.stack([values.real, values.imag], axis=1), factor
