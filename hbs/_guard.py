"""重建合法性门：识别「有限、无异常，但不是一条曲线」的退化重建。

## 为什么需要

`reconstruct_from_hbs` 的近临界输入（大量面片 `|μ| → 1`）会让
`geodesic_welding` 把所有外点压到 `|z| = 1`：输出是 `circle_num` 个完全相同的
`1.0`，无异常、无 NaN、`extent = 0`。调用方拿到的是「一个点」而不是曲线，
下游的保真度、配准、绘图全部失去意义，而且**没有任何信号**。

这道门只认「有非零展布」的边界，因此同时覆盖常数哨兵与全部重合两种退化。

## 与回退阶梯的关系

`get_hbs(..., validate_reconstruction=True)` 在默认网格上先跑一次，门不过就换
`circle = circle_point_num // 2` 再跑一次，两级都不过则抛 `RuntimeError`——
不回退成「未校验的结果」。实测（研究仓 shape100-v1，冻结 100 例）：回退触发 4 例、
救回 1 例（`real_chopper-3` 式近临界），`concave_quad`/`real_Bone-3`/
`real_hammer-3` 两级皆为哨兵。

门默认**关闭**：它要额外花掉一次完整重建（LSQC 求解），对正常形状是纯开销，
且会改变「同一调用在默认网格上必定返回」的既有语义。

来源：研究仓 1.2.0 vendored 角速度路径的 `_reconstruct_is_valid`
（`src/hbs/angular.py`，2026-10-24 修复回退结果未复检的 bug），2.3.2 移入库。
"""

from __future__ import annotations

import numpy as np

#: 判定「退化为一点」的展布阈值：`max|z − mean(z)|` 必须大于它。
#: 与 1.2.0 vendored 实现取同一数值，保证迁移前后判定逐例一致。
DEGENERATE_EXTENT = 1e-6

#: 回退阶梯的源圆下限：`circle_point_num // 2` 不会再低于它。
MIN_FALLBACK_CIRCLE = 64


def reconstruction_is_valid(mu: np.ndarray, disk) -> bool:
    """`reconstruct_from_hbs(mu, disk)` 是否给出了一条非退化边界。

    :param `mu`: `disk.face_num` 个复数（Beltrami 系数）
    :param `disk`: 与之配套的 `DiskMesh`
    :return: `True` 表示边界有限、形状正确且展布大于 `DEGENERATE_EXTENT`

    重建本身抛异常时返回 `False`（近临界输入的一种失败形态），由调用方决定
    是回退还是报错。
    """
    from hbs.hbs import reconstruct_from_hbs

    try:
        boundary, *_ = reconstruct_from_hbs(np.asarray(mu), disk)
    except Exception:
        return False
    boundary = np.asarray(boundary, float)
    if boundary.ndim != 2 or boundary.shape != (int(disk.circle_num), 2):
        return False
    z = boundary[:, 0] + 1j * boundary[:, 1]
    if not np.all(np.isfinite(z)):
        return False
    return float(np.max(np.abs(z - z.mean()))) > DEGENERATE_EXTENT
