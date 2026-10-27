"""相似不变评估（fidelity_sim）—— Phase 1 落地。

主指标 fid_sim：去质心 → 面积归一 → 最优旋转（含镜像）→ RMS / R_eff。
解决 HBS 尺度不变性（landmark 强制 x 半轴=1，面积比=1/rx²）导致的裸 r_min 误判。

单点 r_min / 面积比 裸指标不再作主判据，仅作辅指标解释。
"""

import numpy as np


def _shoelace_area(z):
    z = np.asarray(z)
    if z.ndim == 2:
        z = z[:, 0] + 1j * z[:, 1]
    zc = np.concatenate([z, z[:1]])
    return 0.5 * np.abs(np.imag(np.conj(zc[:-1]) * zc[1:])).sum()


def _resample(z, n=400):
    z = np.asarray(z, dtype=complex)
    z = np.concatenate([z, z[:1]])
    d = np.abs(np.diff(z))
    s = np.concatenate([[0], np.cumsum(d)])
    total = s[-1]
    if total <= 0 or not np.isfinite(total):
        return z[:-1]
    s = s / total
    return np.interp(np.linspace(0, 1, n, endpoint=False), s, z, period=1.0)


def fidelity_sim(rec, true, n=400) -> float:
    """相似不变保真度：面积归一 + 最优旋转/镜像。

    rec, true: (m,2) 或复向量边界。返回 RMS / R_eff，无量纲。
    实测基线（disk 0.01/1000，Phase 1 标定 08-20）：circle 0.0012 / ellipse 1.2×1.0 0.0074
    / ellipse 1.2×0.8 0.019（均 <0.10 健康门）；deep_star |cos5| r=0.2 0.21 / r=0.4 0.14；
    blob/superellipse/peanut 0.28/0.04/0.02（焊接对非椭圆谐波敏感）。健康门 = fid_sim<0.10
    且 post 不劣于 base（Δ<0.01）；深凹以 Δfid_sim 下降为唯一判据。
    """
    r = _resample(np.asarray(rec[:, 0] + 1j * np.asarray(rec[:, 1]) if np.asarray(rec).ndim == 2 else np.asarray(rec, dtype=complex)), n)
    o = _resample(np.asarray(true[:, 0] + 1j * np.asarray(true[:, 1]) if np.asarray(true).ndim == 2 else np.asarray(true, dtype=complex)), n)
    r = r - r.mean()
    o = o - o.mean()
    a_r = _shoelace_area(r)
    a_o = _shoelace_area(o)
    if not np.isfinite(a_r) or not np.isfinite(a_o) or a_r <= 0 or a_o <= 0:
        return float("inf")
    r = r * np.sqrt(a_o / a_r)
    r_eff = np.sqrt(a_o / np.pi)
    if not np.isfinite(r_eff) or r_eff == 0:
        return float("inf")

    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        # 退化：离散网格
        best = float("inf")
        for phi in np.linspace(0, 2 * np.pi, 72, endpoint=False):
            for refl in (False, True):
                rr = np.conj(r) * np.exp(1j * phi) if refl else r * np.exp(1j * phi)
                e = float(np.sqrt(np.mean(np.abs(rr - o) ** 2)))
                if e < best:
                    best = e
        return best / r_eff

    best = float("inf")
    for refl in (False, True):
        rr_base = np.conj(r) if refl else r

        def err(phi):
            rr = rr_base * np.exp(1j * phi)
            return float(np.sqrt(np.mean(np.abs(rr - o) ** 2)))

        # 72 网格初筛 + 连续精修
        phis = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        phi0 = phis[np.argmin([err(p) for p in phis])]
        # 在 phi0 邻域精修
        lo = phi0 - np.pi / 36
        hi = phi0 + np.pi / 36
        res = minimize_scalar(err, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        if res.fun < best:
            best = res.fun
        # 也做全局精修兜底
        res2 = minimize_scalar(err, bounds=(0, 2 * np.pi), method="bounded", options={"xatol": 1e-4})
        if res2.fun < best:
            best = res2.fun
    return best / r_eff


def depth_rel(z) -> float:
    """相对谷深 (Rmax-Rmin)/Rmean，面积归一前原始相对比值，仅辅指标。"""
    z = np.asarray(z)
    if z.ndim == 2:
        z = z[:, 0] + 1j * z[:, 1]
    r = np.abs(z - z.mean())
    m = float(r.mean())
    if m == 0 or not np.isfinite(m):
        return 0.0
    return float((r.max() - r.min()) / m)


def regularity(z):
    """几何合法性：极值数、最大曲率、是否自交（近似）。"""
    z = np.asarray(z, dtype=float)
    if z.ndim == 2:
        pts = z
    else:
        pts = np.column_stack([z.real, z.imag])
    n = len(pts)
    r = np.abs(pts[:, 0] + 1j * pts[:, 1] - np.mean(pts[:, 0] + 1j * pts[:, 1]))
    c = np.concatenate([r, r[:1]])
    n_min = int(((c[1:-1] < c[:-2]) & (c[1:-1] < c[2:])).sum())
    n_max = int(((c[1:-1] > c[:-2]) & (c[1:-1] > c[2:])).sum())
    # 最大曲率近似（离散转角/边长）
    curv = np.zeros(n)
    for i in range(n):
        p0, p, p1 = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = p - p0, p1 - p
        n1, n2 = np.hypot(*v1), np.hypot(*v2)
        if n1 * n2 == 0:
            continue
        dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        ang = np.arccos(dot)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:
            ang = -ang
        curv[i] = ang / ((n1 + n2) / 2 + 1e-12)
    return dict(n_min=n_min, n_max=n_max, max_abs_curv=float(np.max(np.abs(curv))) if len(curv) else 0.0)
