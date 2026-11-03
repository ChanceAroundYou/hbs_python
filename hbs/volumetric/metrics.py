"""三维离散畸变量：奇异值、Jacobian 与 K。

对单个 P1 四面体，设 ``A = DG`` 为仿射部分的 3×3 矩阵，``s1 ≥ s2 ≥ s3`` 为其
奇异值，``J = det A``。定义

    K_O = s1³ / |J|,        K_I = |J| / s3³,        K = max(K_O, K_I)

``K = 1`` 当且仅当该单元是相似变换（旋转 + 均匀缩放）；纯转动与均匀缩放都不
产生畸变。这个量是**几何**的，不含材料参数，因而不是应力或应变。
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .mesh import TetraMesh, as_tetramesh


def deformation_data(
    values: np.ndarray, mesh: TetraMesh | Mapping
):
    """返回 ``(A, U, s, Vh, J, K_O, K_I)``，逐四面体。

    ``A`` 是 3×3 仿射矩阵，``U, s, Vh`` 是 SVD，``J`` 是行列式。
    奇异值下限截到 ``1e-14``、``|J|`` 下限截到 ``1e-30``，避免翻折单元产生
    除零或 ``inf``；此时读数是饱和值而不是真实值，调用方应同时看 ``min_J``。
    """
    mesh = as_tetramesh(mesh)
    values = np.asarray(values, dtype=float)
    if values.shape != mesh.points.shape:
        raise ValueError(
            f"values must have shape {mesh.points.shape}, got {values.shape}"
        )

    A = np.einsum("tvi,tvj->tij", values[mesh.tetra], mesh.gradients)
    U, s, Vh = np.linalg.svd(A)
    J = np.linalg.det(A)
    absJ = np.maximum(np.abs(J), 1e-30)
    sigma = np.maximum(s, 1e-14)
    KO = sigma[:, 0] ** 3 / absJ
    KI = absJ / sigma[:, -1] ** 3
    return A, U, sigma, Vh, J, KO, KI


def element_k(values: np.ndarray, mesh: TetraMesh | Mapping) -> np.ndarray:
    """逐四面体的 ``K = max(K_O, K_I)``（长度 = 单元数）。"""
    *_, KO, KI = deformation_data(values, mesh)
    return np.maximum(KO, KI)


def distortion_metrics(values: np.ndarray, mesh: TetraMesh | Mapping) -> dict:
    """汇总畸变读数。

    ``weighted_mean_K`` / ``weighted_p95_K`` 以**源域**四面体体积加权，
    因此与网格疏密无关；``max_K`` 是未加权的逐单元最大值。
    """
    mesh = as_tetramesh(mesh)
    _, _, _, _, J, KO, KI = deformation_data(values, mesh)
    K = np.maximum(KO, KI)

    volume = mesh.volume
    total = float(np.sum(volume))
    if total <= 0:
        raise ValueError("mesh has non-positive total volume")
    order = np.argsort(K)
    cumulative = np.cumsum(volume[order]) / total
    return dict(
        max_K=float(np.max(K)),
        min_J=float(np.min(J)),
        negative_or_zero_tets=int(np.sum(J <= 0)),
        weighted_mean_K=float(np.average(K, weights=volume)),
        weighted_p95_K=float(np.interp(0.95, cumulative, K[order])),
    )


def boundary_vertex_k(values: np.ndarray, mesh: TetraMesh | Mapping) -> np.ndarray:
    """把逐单元 K 按体积加权平均到边界顶点（顺序与 ``mesh.points[interior:]`` 一致）。

    用于在原始表面网格上着色；它只是逐单元 K 的邻域平均，不是在该表面上重新
    求解的连续 K。没有相邻单元的顶点取 1.0。
    """
    mesh = as_tetramesh(mesh)
    K = element_k(values, mesh)
    n = mesh.interior_count
    count = len(mesh.points) - n
    weight = np.zeros(count)
    total = np.zeros(count)
    for local in range(4):
        node = mesh.tetra[:, local] - n
        valid = node >= 0
        np.add.at(weight, node[valid], mesh.volume[valid])
        np.add.at(total, node[valid], mesh.volume[valid] * K[valid])
    return np.where(weight > 1e-12, total / np.maximum(weight, 1e-300), 1.0)
