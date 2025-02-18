import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator


def smooth_resample(A, m=None):
    """
    对边界点矩阵A进行光滑均匀重采样，生成m个点的矩阵B
    :param A: nx2矩阵，原始边界点坐标
    :param m: 重采样后的点数
    :return: mx2矩阵，重采样后的边界点
    """
    if m is None:
        m = A.shape[0]

    # 检查闭合性
    is_closed = np.all(np.abs(A[0, :] - A[-1, :]) < 1e-8)
    if not is_closed:
        A = np.vstack([A, A[0, :]])  # 非闭合则闭合处理

    # 计算累积弧长
    diffs = np.diff(A, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]

    # 样条插值设置
    t = cum_dists
    x = A[:, 0]
    y = A[:, 1]

    if is_closed:
        # 周期性三次样条插值（闭合曲线）
        ppx = CubicSpline(t, x, bc_type="periodic")
        ppy = CubicSpline(t, y, bc_type="periodic")
    else:
        # 保形分段三次插值（非闭合曲线）
        ppx = PchipInterpolator(t, x)
        ppy = PchipInterpolator(t, y)

    # 生成均匀参数
    if is_closed:
        t_new = np.linspace(0, total_length, m + 2)
        t_new = t_new[:-1]  # 避免重复首尾点
    else:
        t_new = np.linspace(0, total_length, m + 1)

    # 计算新点
    x_new = ppx(t_new)
    y_new = ppy(t_new)
    B = np.column_stack([x_new, y_new])
    B = B[:-1, :]
    return B
