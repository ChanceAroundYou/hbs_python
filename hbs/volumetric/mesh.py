"""四面体网格容器与通用构造工具。

约定：内部顶点排在前面（``[:interior_count]``），边界顶点排在后面
（``[interior_count:]``）。这与二维 ``hbs.mesh`` 的「圆盘点在前」不是同一套顺序，
三维求解器依赖这条约定来把边界迹固定在末尾。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

_CORE_FIELDS = ("points", "tetra", "gradients", "volume", "interior_count")
_OPTIONAL_FIELDS = ("trace", "meta")


@dataclass
class TetraMesh:
    """三维单纯复形网格。

    y 方向与二维 ``Mesh`` 不同：这里存的是**源域**顶点 ``points``（n×3）、
    四面体索引 ``tetra``（m×4）、每个四面体上 P1 基函数的梯度 ``gradients``
    （m×4×3）、有符号体积 ``volume``（m）与内部顶点数 ``interior_count``。

    ``trace`` 是目标侧边界迹（只对边界顶点定义），``meta`` 供调用方携带
    自己的单元标签；其它键放进 ``extra``，因此仍可按字典方式索引：

    >>> mesh["points"].shape, mesh["stations"].shape  # doctest: +SKIP
    """

    points: np.ndarray
    tetra: np.ndarray
    gradients: np.ndarray
    volume: np.ndarray
    interior_count: int
    trace: np.ndarray | None = None
    meta: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        self.tetra = np.asarray(self.tetra, dtype=np.int64)
        self.gradients = np.asarray(self.gradients, dtype=float)
        self.volume = np.asarray(self.volume, dtype=float)
        self.interior_count = int(self.interior_count)

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (vertices, 3)")
        if self.tetra.ndim != 2 or self.tetra.shape[1] != 4:
            raise ValueError("tetra must have shape (elements, 4)")
        if self.gradients.shape != (len(self.tetra), 4, 3):
            raise ValueError("gradients must have shape (elements, 4, 3)")
        if self.volume.shape != (len(self.tetra),):
            raise ValueError("volume must have shape (elements,)")
        if not 0 <= self.interior_count <= len(self.points):
            raise ValueError("interior_count must lie in [0, vertices]")
        if len(self.tetra) and (self.tetra.min() < 0 or self.tetra.max() >= len(self.points)):
            raise ValueError("tetra indices out of range")

    def __getitem__(self, key: str) -> Any:
        if key in _CORE_FIELDS or key in _OPTIONAL_FIELDS:
            return getattr(self, key)
        return self.extra[key]

    def __contains__(self, key: object) -> bool:
        return (
            key in _CORE_FIELDS
            or key in _OPTIONAL_FIELDS
            or key in self.extra
        )

    def keys(self) -> tuple[str, ...]:
        return (*_CORE_FIELDS, *_OPTIONAL_FIELDS, *self.extra)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def as_dict(self) -> dict[str, Any]:
        """展开成普通字典；与研究仓历史脚本的字典接口兼容。"""
        return {key: self[key] for key in self.keys()}


def as_tetramesh(mesh: TetraMesh | Mapping[str, Any]) -> TetraMesh:
    """把 ``TetraMesh`` 或历史字典统一成一个 ``TetraMesh`` 视图。

    字典缺少 ``gradients`` / ``volume`` 时就地重算；已有值则原样复用，因此
    对同一网格反复调用不会改变读数。
    """
    if isinstance(mesh, TetraMesh):
        return mesh
    if not isinstance(mesh, Mapping):
        raise TypeError("mesh must be a TetraMesh or a mapping")

    missing = [key for key in ("points", "tetra", "interior_count") if key not in mesh]
    if missing:
        raise ValueError(f"mesh is missing required keys: {', '.join(missing)}")

    points = np.asarray(mesh["points"], dtype=float)
    tetra = np.asarray(mesh["tetra"], dtype=np.int64)
    interior_count = int(mesh["interior_count"])
    if "gradients" in mesh and "volume" in mesh:
        gradients = np.asarray(mesh["gradients"], dtype=float)
        volume = np.asarray(mesh["volume"], dtype=float)
    else:
        gradients, volume = element_gradients(points, tetra)

    known = set(_CORE_FIELDS) | set(_OPTIONAL_FIELDS)
    extra = {key: value for key, value in mesh.items() if key not in known}
    return TetraMesh(
        points,
        tetra,
        gradients,
        volume,
        interior_count,
        trace=mesh.get("trace"),
        meta=mesh.get("meta"),
        extra=extra,
    )


def element_gradients(
    points: np.ndarray, tetra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """P1 四面体的四个基函数梯度与有符号体积。

    第 v 个基函数对应顶点 ``tetra[:, v]``，梯度满足 ``∇N_v · (x_j - x_0) = δ_vj``。
    """
    points = np.asarray(points, dtype=float)
    tetra = np.asarray(tetra, dtype=np.int64)
    matrix = np.transpose(
        points[tetra[:, 1:]] - points[tetra[:, 0, None]], (0, 2, 1)
    )
    inverse = np.linalg.inv(matrix)
    gradients = np.concatenate(
        [-np.sum(inverse, axis=1)[:, None, :], inverse], axis=1
    )
    volume = np.linalg.det(matrix) / 6.0
    return gradients, volume


def build_tetramesh(
    points: np.ndarray,
    tetra: np.ndarray,
    *,
    interior_count: int,
    trace: np.ndarray | None = None,
    orient: bool = True,
) -> TetraMesh:
    """由顶点与四面体索引构造 ``TetraMesh``。

    ``orient=True``（默认）时逐个把负体积单元翻正；翻正后仍非正的单元说明网格
    退化，直接报错而不是带着翻折单元继续求解。
    """
    points = np.asarray(points, dtype=float)
    tetra = np.asarray(tetra, dtype=np.int64).copy()
    if len(tetra) and (tetra.min() < 0 or tetra.max() >= len(points)):
        raise ValueError("tetra indices out of range")

    if orient and len(tetra):
        matrix = np.transpose(
            points[tetra[:, 1:]] - points[tetra[:, 0, None]], (0, 2, 1)
        )
        flip = np.linalg.det(matrix) < 0
        tetra[flip, 1], tetra[flip, 2] = (
            tetra[flip, 2].copy(),
            tetra[flip, 1].copy(),
        )

    # 先查退化单元再求逆：奇异矩阵会让 np.linalg.inv 抛 LinAlgError，
    # 这里给出带单元数的明确错误。
    if len(tetra):
        matrix = np.transpose(
            points[tetra[:, 1:]] - points[tetra[:, 0, None]], (0, 2, 1)
        )
        bad = int(np.sum(np.linalg.det(matrix) <= 0))
        if bad:
            raise ValueError(
                f"{bad} tetrahedra have non-positive volume; the mesh is degenerate"
            )
    gradients, volume = element_gradients(points, tetra)
    return TetraMesh(points, tetra, gradients, volume, interior_count, trace=trace)


def ball_tetra_mesh(
    boundary_count: int = 48, interior_count: int = 10, seed: int = 0
) -> TetraMesh:
    """单位球内的一层随机四面体网格（确定性，``seed`` 固定）。

    边界点取 Fibonacci 球面并施加一个随机旋转；内部点按 ``r^(1/3)`` 分布，
    原点是第一个内部点。外部用 Delaunay 连接，再把负体积单元翻正。
    """
    if boundary_count < 4:
        raise ValueError("boundary_count must be at least 4")
    if interior_count < 1:
        raise ValueError("interior_count must be at least 1")

    rng = np.random.default_rng(seed)
    index = np.arange(boundary_count)
    zz = 1 - 2 * (index + 0.5) / boundary_count
    phi = np.pi * (3 - np.sqrt(5)) * index
    boundary = np.c_[
        np.sqrt(1 - zz**2) * np.cos(phi),
        np.sqrt(1 - zz**2) * np.sin(phi),
        zz,
    ]
    boundary = Rotation.random(random_state=rng).apply(boundary)

    interior = rng.normal(size=(interior_count, 3))
    interior /= np.linalg.norm(interior, axis=1)[:, None]
    interior *= (0.68 * rng.random(interior_count) ** (1 / 3))[:, None]
    interior[0] = 0.0

    points = np.r_[interior, boundary]
    tetra = Delaunay(points).simplices.copy()
    return build_tetramesh(points, tetra, interior_count=interior_count)
