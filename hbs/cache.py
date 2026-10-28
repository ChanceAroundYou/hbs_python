"""盘网格的**文件缓存**：避免每次重建 Delaunay 与 FEM 算子。

`get_unit_disk(density, circle)` 在密网格上是主要成本：`Delaunay` 197 ms、
`DiskMesh.set_operator` 126 ms（向量化后），合计约 210 ms；而求解本身在 0.1 s
量级。缓存把这两项都变成一次磁盘读。

## 缓存什么

- `vert` / `face`：网格本身。Delaunay 的随机构造是整个流程里最贵的一步。
- FEM 的 `A` / `B`（弱拉普拉斯块）：11.5 ms 构造，但**依赖 vert/face**，与形状
  无关，所以跟着网格一起缓存。
- **不缓存 `splu` 分解**：`SuperLU` 对象不可 pickle（`TypeError: cannot pickle
  'SuperLU' object`），退而存 `L`/`U`/`perm` 要 34.5 MB 且重建不省时间。分解
  68.6 ms，是缓存后剩下的主要成本，见下面第 3 条。

## 文件布局

    ~/.cache/hbs/mesh/d<density>_c<circle>_v<schema>/     # 默认根
      meta.json        # density/circle/vert_num/face_num/schema/哈希
      mesh.npz         # vert, face
      fem.npz          # A（csr 三元组）, B

`HBS_CACHE_DIR` 可覆盖根目录；`HBS_CACHE=0`（或 `"false"`/`"off"`）禁用缓存。
写入用临时目录 + 原子 rename，两个进程同时算同一网格不会互相破坏。

## 正确性

缓存命中后返回的网格与现场计算**逐位相同**（`np.array_equal`）；`tests/`
里有对应用例。缓存文件带 schema 版本与形状字段的校验，不匹配即重算。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

#: 缓存格式版本。改动 `DiskMesh`/FEM 算子的构造方式时递增，旧缓存自动失效。
SCHEMA_VERSION = 1

_ENV_DIR = "HBS_CACHE_DIR"
_ENV_ENABLE = "HBS_CACHE"


def cache_root() -> Path:
    """缓存根目录（`HBS_CACHE_DIR` 或 `~/.cache/hbs`）。"""
    override = os.environ.get(_ENV_DIR)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "hbs"


def cache_enabled() -> bool:
    value = os.environ.get(_ENV_ENABLE, "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def disk_cache_dir(density: float, circle_point_num: int) -> Path:
    key = f"d{density:g}_c{int(circle_point_num)}_v{SCHEMA_VERSION}"
    return cache_root() / "mesh" / key


def _mesh_digest(vert: np.ndarray, face: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vert, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(face, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _fem_operator(disk):
    """弱拉普拉斯块 `(A, B)`，与 `_angular.joint` 用的是同一份定义。

    `A`` = 内部节点 × 内部节点，`B` = 内部节点 × 边界节点。做成这里的一份是为了
    缓存层不依赖 `_angular`（zipper 也要用同一个 FEM 延拓）。
    """
    from scipy.sparse import diags

    area = np.abs(np.asarray(disk.area, dtype=float))
    stiffness = (
        disk.Dx.T @ diags(area) @ disk.Dx + disk.Dy.T @ diags(area) @ disk.Dy
    ).tocsr()
    mass = np.bincount(
        disk.face.ravel(), weights=np.repeat(area / 3, 3), minlength=disk.vert_num
    )
    ii = np.arange(disk.circle_num, disk.vert_num)
    bd = np.arange(disk.circle_num)
    weak = diags(1 / np.sqrt(np.pi * mass[ii])) @ stiffness[ii]
    return weak[:, ii].tocsr(), weak[:, bd].tocsr()


def _save_sparse(path: Path, name: str, mat: csr_matrix) -> None:
    np.savez(
        path,
        **{
            f"{name}_data": mat.data,
            f"{name}_indices": mat.indices,
            f"{name}_indptr": mat.indptr,
            f"{name}_shape": np.asarray(mat.shape, dtype=np.int64),
        },
    )


def _load_sparse(pack, name: str) -> csr_matrix:
    return csr_matrix(
        (pack[f"{name}_data"], pack[f"{name}_indices"], pack[f"{name}_indptr"]),
        shape=tuple(int(x) for x in pack[f"{name}_shape"]),
    )


def load_unit_disk(density: float, circle_point_num: int):
    """取单位盘网格与 FEM 块；命中缓存则读盘，否则计算并写入。

    返回 `(disk, fem)`，`fem = (A, B)`。
    """
    from hbs.mesh import get_unit_disk  # 延迟导入，避免循环

    directory = disk_cache_dir(density, circle_point_num)
    meta_path = directory / "meta.json"
    mesh_path = directory / "mesh.npz"
    fem_path = directory / "fem.npz"

    if cache_enabled() and meta_path.exists() and mesh_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("schema") != SCHEMA_VERSION:
                raise ValueError("schema mismatch")
            with np.load(mesh_path) as pack:
                vert = np.array(pack["vert"], dtype=np.float64)
                face = np.array(pack["face"], dtype=np.int64)
            if meta.get("digest") != _mesh_digest(vert, face):
                raise ValueError("digest mismatch")
            disk = _disk_from_arrays(vert, face, circle_point_num)
            fem = None
            if fem_path.exists():
                with np.load(fem_path) as pack:
                    fem = (_load_sparse(pack, "A"), _load_sparse(pack, "B"))
            if fem is None:
                fem = _fem_operator(disk)
                _write_fem(fem_path, fem)
            return disk, fem
        except Exception:  # noqa: BLE001 - 缓存坏了就重算，不让它变成硬故障
            shutil.rmtree(directory, ignore_errors=True)

    disk = get_unit_disk(density, circle_point_num)
    fem = _fem_operator(disk)
    if cache_enabled():
        _write_cache(directory, disk, fem, density, circle_point_num)
    return disk, fem


def _disk_from_arrays(vert: np.ndarray, face: np.ndarray, circle_point_num: int):
    """由 `vert`/`face` 重建 `DiskMesh`（绕过 Delaunay 与 `get_unit_disk`）。"""
    from hbs.mesh import DiskMesh

    nc = int(circle_point_num)
    return DiskMesh(
        circle=np.array(vert[:nc], dtype=np.float64),
        in_vert=np.array(vert[nc:], dtype=np.float64),
        out_vert=None,
        face=np.array(face, dtype=np.int64),
    )


def _write_fem(path: Path, fem) -> None:
    A, B = fem
    tmp = path.with_suffix(".tmp.npz")
    _save_sparse_tmp(tmp, A, B)
    os.replace(tmp, path)


def _save_sparse_tmp(path: Path, A, B) -> None:
    np.savez(
        path,
        A_data=A.data, A_indices=A.indices, A_indptr=A.indptr,
        A_shape=np.asarray(A.shape, dtype=np.int64),
        B_data=B.data, B_indices=B.indices, B_indptr=B.indptr,
        B_shape=np.asarray(B.shape, dtype=np.int64),
    )


def _write_cache(directory: Path, disk, fem, density, circle) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".tmp-"))
    try:
        vert = np.asarray(disk.vert, dtype=np.float64)
        face = np.asarray(disk.face, dtype=np.int64)
        np.savez(tmp / "mesh.npz", vert=vert, face=face)
        _save_sparse_tmp(tmp / "fem.npz", *fem)
        meta = {
            "schema": SCHEMA_VERSION,
            "density": float(density),
            "circle_point_num": int(circle),
            "vert_num": int(vert.shape[0]),
            "face_num": int(face.shape[0]),
            "digest": _mesh_digest(vert, face),
        }
        (tmp / "meta.json").write_text(
            json.dumps(meta, indent=1), encoding="utf-8"
        )
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        os.replace(tmp, directory)
    except Exception:  # noqa: BLE001 - 写缓存失败不应影响求解
        shutil.rmtree(tmp, ignore_errors=True)


def fem_extension_operator(disk, fem=None):
    """返回可直接 `solve` 的 FEM 延拓算子（对 `A` 做一次 LU 分解）。

    `fem` 省略时就地计算 `(A, B)`。`A` 的 LU 分解不可序列化，所以每次进程都要
    重做一次 —— 这是缓存之后剩下的唯一固定开销（密网格约 69 ms）。
    """
    if fem is None:
        fem = _fem_operator(disk)
    A, _B = fem
    return splu(A.tocsc())
