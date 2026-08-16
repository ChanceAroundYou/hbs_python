"""尖角增强：修复 LSQC（P1 元）在 |μ|→1 退化区的面积塌缩。

问题：Beltrami 方程在 |μ|→1（形状尖角处）退化为抛物，P1 元无法表示
代数奇异性 f(z)≈(z-z0)^α → 退化区映射面积塌缩 800× → 尖角（爪/嘴）丢失。

增强（两处）：
1. analytic_extend：退化区用解析奇异解 u0 + A·(z-z0)^α 填充（由周围
   非退化边界顶点最小二乘拟合 u0、A），恢复尖角几何。
2. angle_resample_boundary：焊接前对 LSQC 边界按角度均匀重采样，消除
   微小角度回绕（zipper 输入前提），否则焊接塌缩成单点。

实测：多样本尖角保留 13.3°→25~57°（改善 2-4 倍）。
"""
import numpy as np


def detect_singularities(disk, mu, r_thr=0.99, gap_deg=10):
    """退化面片（|μ|>r_thr）角度聚类 → 奇异点位置（圆盘边界）与 α 估计。

    α 从 |μ|(r) 拟合：|μ| = (α²r^(2α-2)-1)/(α²r^(2α-2)+1)，
    变换 y=log((1+μ)/(1-μ)) = log(α²)+(2α-2)log(r)，斜率 k=2α-2。
    """
    fc = disk.face_center
    abs_mu = np.abs(mu)
    hi = abs_mu > r_thr
    if hi.sum() < 5:
        return []
    ang = np.arctan2(fc[hi, 1], fc[hi, 0])
    a = np.sort(np.degrees(np.mod(ang, 2 * np.pi)))
    groups, cur = [], [a[0]]
    for i in range(1, len(a)):
        if a[i] - a[i - 1] > gap_deg:
            groups.append(cur)
            cur = [a[i]]
        else:
            cur.append(a[i])
    groups.append(cur)
    if len(groups) > 1 and (a[0] + 360 - a[-1]) < gap_deg:
        groups[0] = groups[-1] + groups[0]
        groups = groups[:-1]

    sing = []
    for g in groups:
        th0 = np.radians(np.mean(g) % 360)
        z0 = np.array([np.cos(th0), np.sin(th0)])
        d = np.linalg.norm(fc[hi] - z0, axis=1)
        near = d < 0.15
        if near.sum() > 5:
            rr = d[near]
            mm = abs_mu[hi][near]
            y = np.log((1 + mm) / (1 - mm))
            x = np.log(rr)
            k, _ = np.polyfit(x, y, 1)
            alpha = float(np.clip((k + 2) / 2, 0.15, 0.99))
        else:
            alpha = 0.5
        sing.append((z0, alpha))
    return sing


def analytic_extend(disk, mu, u_std, singularities, r_enrich=0.16):
    """退化区用解析奇异解 u0 + A·(z-z0)^α 填充。

    对每个奇异点：退化面片（距 z0<r_enrich 且 |μ|>0.9）的顶点，
    用周围非退化边界顶点最小二乘拟合的解析解替换。
    """
    if not singularities:
        return u_std
    V = disk.vert_num
    vert = disk.vert
    face = disk.face
    fc = disk.face_center
    abs_mu = np.abs(mu)
    u_new = u_std.copy()

    zc = vert[:, 0] + 1j * vert[:, 1]
    for z0, alpha in singularities:
        d_f = np.linalg.norm(fc - z0, axis=1)
        degen_faces = (d_f < r_enrich) & (abs_mu > 0.9)
        degen_verts = np.unique(face[degen_faces])
        if len(degen_verts) < 5:
            continue
        nondeg_faces = ~degen_faces
        boundary_verts = np.unique(face[nondeg_faces])
        boundary_verts = boundary_verts[np.isin(boundary_verts, degen_verts)]
        if len(boundary_verts) < 3:
            continue

        # 解析基 ψ(z) = (z-z0)^α（复值）
        z0c = z0[0] + 1j * z0[1]
        dz = zc - z0c
        d = np.abs(dz)
        theta = np.angle(dz)
        psi = d ** alpha * np.exp(1j * alpha * theta)
        # 边界顶点拟合 u = u0 + A·[Re ψ, Im ψ]
        pb = np.stack([psi[boundary_verts].real, psi[boundary_verts].imag], 1)
        ub = u_std[boundary_verts]
        X = np.hstack([np.ones((len(boundary_verts), 1)), pb])
        coef, *_ = np.linalg.lstsq(X, ub, rcond=None)
        u0, A = coef[0], coef[1:]
        # 填充退化顶点
        pv = np.stack([psi[degen_verts].real, psi[degen_verts].imag], 1)
        u_new[degen_verts] = u0[None] + pv @ A
    return u_new


def angle_resample_boundary(s, n=None):
    """焊接前对边界按角度均匀重采样，消除微小角度回绕（zipper 输入前提）。

    LSQC 解出的边界 s 因退化区畸变有 ~1e-3 rad 角度回绕，直接焊接会
    塌缩成单点。按角度均匀重采样（r 插值）恢复单调合法参数化。
    """
    s = np.asarray(s)
    n = n or len(s)
    z = s[:, 0] + 1j * s[:, 1]
    r = np.abs(z)
    th = np.unwrap(np.angle(z))
    th_new = np.linspace(th[0], th[-1], n)
    z2 = np.interp(th_new, th, r) * np.exp(1j * th_new)
    return np.stack([z2.real, z2.imag], 1)
