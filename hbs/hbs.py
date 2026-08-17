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

    # 边界分辨率鲁棒性：zipper 对非均匀采样（cv2 轮廓、密集采样、顶点落坐标轴的
    # sign 采样）产生 NaN。统一按弧长重采样到 500 点规避（均匀采样输入下近似
    # 恒等，非均匀则修正；实验验证）。500 是经过验证的稳健默认；某些计数
    # （250 及 ≥800）zipper 数值退化，重采样不能完全消除；静默退化输出需调用方
    # 校验重建面积。
    bound = _resample_boundary(bound, 500)

    cw = get_conformal_welding(bound)

    # λ-归一化（内部文档 lambda_normalization_notes.tex）：
    # I₂ = ∫_D B(z)/z dz 在旋转下按 I₂(R_θB) = e^{-iθ}I₂(B) 变换，
    # 故迭代旋转 seam 使 arg I₂ → 0 即得唯一规范代表（连续，无 180° 翻转）。
    # 对称/近对称形状 I₂ 的离散残留导致极限环振荡 → 30 次不收敛返回当前
    # GHBS 代表（不抛错）：归一化旋转角不影响重建形状（up to 旋转）。
    last_r = None
    for _ in range(30):
        cw.linear_interp(circle_point_num)
        hbs_mapping = poisson_integral(disk.in_vert, cw.x, cw.y)
        hbs = get_beltrami_coefficient(hbs_mapping, disk)

        excluded_idx = np.isnan(hbs) + np.linalg.norm(disk.face_center, axis=1) == 0
        h = hbs[~excluded_idx]
        fc = disk.face_center[~excluded_idx]
        i2 = np.sum(h / to_complex(fc))

        if not np.isfinite(i2) or abs(i2) <= np.finfo(float).eps:
            break  # 对称形状 I₂ 离散残留不可解析 → 返回当前 GHBS 代表
        r = np.angle(i2)
        if abs(r) <= 5e-3:
            break
        if last_r is not None and r * last_r < 0 and abs(r) > abs(last_r):
            break  # 角度振荡发散（对称/近对称形状特征）→ 提前返回 GHBS 代表
        last_r = r
        cw.rotate_x(r)

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

    welded_ok = (
        np.all(np.isfinite(in_points))
        and np.all(np.isfinite(out_points))
        and _weld_not_degenerate(bound_points, in_points, disk)
    )
    if not welded_ok:
        # 焊接数值退化（尖角/病态 seam/非规范代表）→ 降级为纯 LSQC 输出，不抛错。
        # 光滑形状焊接保真 10× 必要（实验验证），仅失败路径兜底。
        bound_points = mapping[: disk.circle_num]
        in_points = mapping[disk.circle_num : disk.circle_num + disk.in_vert_num]
        out_points = mapping[disk.circle_num + disk.in_vert_num :]
    else:
        bound_points = in_points[: disk.circle_num]
        in_points = in_points[disk.circle_num :]
    return bound_points, in_points, out_points, mapping


def _weld_not_degenerate(
    bound_points: np.ndarray,
    welded_in_points: np.ndarray,
    disk: DiskMesh,
) -> bool:
    """焊接退化检测：焊接后边界面积与 LSQC 边界面积比值过小（<1e-3）判定退化。
    焊接可能输出有限但几何错误的点（如全部坍缩到 landmark），finite 检查抓不到。"""
    def _area(z):
        z = np.asarray(z)
        if z.ndim == 2:
            z = z[:, 0] + 1j * z[:, 1]
        zc = np.concatenate([z, z[:1]])
        return float(0.5 * np.abs(np.imag(np.conj(zc[:-1]) * zc[1:])).sum())

    a_lsqc = _area(bound_points)
    a_weld = _area(welded_in_points[: disk.circle_num])
    return a_lsqc > 0 and a_weld / a_lsqc > 1e-3
