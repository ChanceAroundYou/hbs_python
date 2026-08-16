import numpy as np

from hbs.conformal_welding import ConformalWelding, get_conformal_welding
from hbs.mesh import DiskMesh, get_unit_disk

from hbs.qc import get_beltrami_coefficient, lsqc_solver
from hbs.utils.geodesic_welding import geodesic_welding
from hbs.utils.poisson import integral as poisson_integral
from hbs.utils.cast import to_complex, to_real


def get_hbs(
    bound: np.ndarray[np.floating],
    circle_point_num: int = 500,
    density: float = 0.01,
    disk: DiskMesh = None,
) -> tuple[
    np.ndarray[np.complexfloating],
    np.ndarray[np.floating],
    ConformalWelding,
    DiskMesh,
]:
    """
    Get Beltrami coefficients from boundary points
    :param `bound`: `n x 2` array, boundary points
    :param `circle_point_num`: number of points on the circle
    :param `density`: density of the mesh
    :param `disk`: DiskMesh object
    :return:
        `hbs`: `n` x 1 complex array, Beltrami coefficients defined on triangles
        `he`: `n` x 1 float array, harmonice extension of `hbs`
        `cw`: ConformalWelding object
        `disk`: DiskMesh object
    """
    if disk is None:
        disk = get_unit_disk(density, circle_point_num)

    # 边界方向规范化：harmonic extension 只在顺时针边界（math 约定 signed area<0）
    # 上产生 |μ|<1 的拟共形场；逆时针输入 → |μ|>1 全盘饱和 → μ 失真。
    # 图轮廓（cv2）与合成形状方向可能不一致，统一翻转为顺时针。
    if _signed_area(bound) > 0:
        bound = bound[::-1]

    # 边界分辨率鲁棒性：过疏边界（<400 点）在 zipper 中大量中途点落虚轴
    # → y 退化压到实线 → y_post_norm 不收敛。均匀重采样到 500 点规避。
    # 已知局限：某些计数（250 及 ≥800）zipper 数值退化，重采样不能完全消除；
    # 500 是经过验证的稳健默认。静默退化输出需调用方校验重建面积。
    if bound.shape[0] < 400:
        bound = _resample_boundary(bound, 500)

    cw = get_conformal_welding(bound)

    r = 0
    for _ in range(30):
        cw.rotate_x(r / 2)
        cw.linear_interp(circle_point_num)
        hbs_mapping = poisson_integral(disk.in_vert, cw.x, cw.y)
        hbs = get_beltrami_coefficient(hbs_mapping, disk)

        excluded_idx = np.isnan(hbs) + np.linalg.norm(disk.face_center, axis=1) == 0
        r = np.angle(np.sum(hbs[~excluded_idx]))

        if abs(r) <= 5e-3:
            he_angle = np.angle(np.sum(hbs * to_complex(disk.face_center)))
            if he_angle < 0 or he_angle == np.pi:
                r += 2 * np.pi
            else:
                break
    else:
        # 近旋转对称形状（Σhbs≈0）归一化振荡不收敛 → 快速抛错而非挂起
        raise RuntimeError("HBS normalization did not converge in 30 iterations")

    return hbs, hbs_mapping, cw, disk


def _signed_area(bound: np.ndarray[np.floating]) -> float:
    """多边形带符号面积（shoelace）。正=逆时针（math 约定），负=顺时针。"""
    x = bound[:, 0]
    y = bound[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _resample_boundary(bound: np.ndarray[np.floating], num: int) -> np.ndarray:
    """按弧长均匀重采样边界到 num 点（闭合多边形）。"""
    z = bound[:, 0] + 1j * bound[:, 1]
    z = np.concatenate([z, z[:1]])
    d = np.abs(np.diff(z))
    s = np.concatenate([[0], np.cumsum(d)])
    s = s / s[-1]
    tt = np.linspace(0, 1, num, endpoint=False)
    z2 = np.interp(tt, s, z, period=1.0)
    return np.stack([z2.real, z2.imag], 1)


def reconstruct_from_hbs(
    hbs: np.ndarray[np.complexfloating], disk: DiskMesh, eps: float = 0.0
):
    """
    Reconstruct original shape from HBS
    :param `hbs`: complex array with length `disk.face_num`, Beltrami coefficients defined on triangles
    :param `disk`: DiskMesh object
    :param `eps`: LSQC 稳定性正则化（尖角处 |mu|→1 病态，>0 改善重建，默认 0）
    :return:
        `bound_points`: `disk.circle_num` x 2 array, boundary points
        `in_points`: `disk.in_vert_num` x 2 array, inner points
        `out_points`: `disk.out_vert_num` x 2 array, outer points
        `mapping`: `disk.vert_num` x 2 array, vertex coordinates of the solved mapping
    """
    assert isinstance(disk, DiskMesh), "`disk` must be DiskMesh object"

    assert isinstance(hbs, np.ndarray) and np.issubdtype(
        hbs.dtype, np.complexfloating
    ), "`hbs` must be complex array"
    assert hbs.ndim == 1 and hbs.shape[0] == disk.face_num, (
        "the length of `hbs` is `n` and `n` is the number of faces"
    )

    target = np.array([[1, 0], [0, 0]], dtype=np.float64)

    one_pos = np.all(disk.vert == target[0], axis=1)
    zero_pos = np.all(disk.vert == target[1], axis=1)
    landmark = np.where(one_pos | zero_pos)[0]

    mapping = lsqc_solver(hbs, landmark, target, disk, eps)
    bound_points = mapping[: disk.circle_num]
    in_points = mapping[: disk.in_vert_num + disk.circle_num]
    out_points = mapping[disk.in_vert_num + disk.circle_num :]

    in_points, out_points = geodesic_welding(
        to_complex(in_points),
        to_complex(out_points),
        to_complex(bound_points),
        to_complex(disk.circle),
    )
    in_points = to_real(in_points)
    out_points = to_real(out_points)

    bound_points = in_points[: disk.circle_num]
    in_points = in_points[disk.circle_num :]
    return bound_points, in_points, out_points, mapping
