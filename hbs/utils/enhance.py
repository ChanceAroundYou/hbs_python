"""尖角增强：压缩 seam 塌缩核恢复尖角（修复 LSQC |μ|→1 退化区面积塌缩）。

问题：Beltrami 方程在 |μ|→1（形状尖角处）退化为抛物，LSQC（P1 元）无法表示
代数奇异性 → 退化区映射面积塌缩 → seam（LSQC 边界映射）在尖角处塌缩成
点簇（局部 blob 极小）→ zipper 焊接输出尖角钝化（爪/嘴丢失）。

原理（实测验证）：
- 焊接输出**只由 seam s 决定**（内部/外环顶点扰动对输出零影响）。
- seam 塌缩弧是输出尖角的来源：压缩塌缩核（向核中心收缩）→ 共形焊接
  产生更尖的 cusp。实测压缩核 0.5 倍：beak 143°→169°、tail 148-177°。
- 关键约束：压缩必须**端点保持不动**（ramp 渐变），否则破坏与肩部连续
  性 → 输出折叠自交；且排除 landmark 固定点（[1,0]）避免伪角。

局限（诚实标注）：压缩引入的 seam 修改非共形一致，部分样本（BW15 等）
自交数略增、chamfer 略回退；sharp corner 数量/角度一致提升但非普适。
"""
import numpy as np


def detect_collapsed_cores(s, blob_thr=0.01, win=30, min_len=20, landmark_idx=0, land_excl=25):
    """检测 seam 上局部 blob 小的连续核（尖角塌缩区）。

    blob = 滑动窗口（win 个连续 seam 点）包围盒对角线。连续 blob < blob_thr
    的索引段即塌缩核。排除 landmark（圆盘边界 [1,0] 顶点）邻域。

    :return: [(lo, hi)] 索引范围列表
    """
    n = len(s)
    blob = np.empty(n)
    half = win // 2
    for i in range(n):
        idxs = np.arange(i - half, i + half + 1) % n
        pts = s[idxs]
        blob[i] = np.linalg.norm(pts.max(0) - pts.min(0))

    m = blob < blob_thr
    clusters = []
    cur = []
    for i in np.where(m)[0]:
        if cur and (i == cur[-1] + 1):
            cur.append(i)
        else:
            if cur:
                clusters.append(cur)
            cur = [i]
    if cur:
        clusters.append(cur)
    # 首尾相接合并（seam 是环）
    if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == n - 1:
        clusters[0] = clusters[-1] + clusters[0]
        clusters = clusters[:-1]

    def near_landmark(sg):
        return any(
            min(abs(i - landmark_idx), abs(i - (landmark_idx + n)), abs(i - (landmark_idx - n))) <= land_excl
            for i in sg
        )

    return [(sg[0], sg[-1]) for sg in clusters if len(sg) >= min_len and not near_landmark(sg)]


def compress_cores(s, cores, factor=0.5):
    """核向核中心压缩，但端点保持不动（ramp：端 0 → 中 factor）。

    端点不动保与肩部连续性（避免输出折叠）；中心压缩最多产生尖 cusp。
    """
    s2 = s.copy()
    n = len(s2)
    for lo, hi in cores:
        idxs = np.arange(lo, hi + 1) % n
        L = len(idxs)
        if L < 4:
            continue
        center = s2[idxs[L // 2]]
        pos = np.abs(np.arange(L) - (L - 1) / 2) / ((L - 1) / 2)
        f = 1 - (1 - factor) * (1 - pos)
        s2[idxs] = center + f[:, None] * (s2[idxs] - center)
    return s2


def sharpen_seam(s, factor=0.5):
    """seam 尖角增强入口：检测塌缩核 + 压缩。返回增强后的 seam。"""
    n = len(s)
    # landmark = 圆盘边界 [1,0] 顶点
    one = np.abs(s[:, 0] - 1) < 1e-9
    lm = np.where(one)[0]
    landmark_idx = int(lm[0]) if len(lm) else 0
    cores = detect_collapsed_cores(s, landmark_idx=landmark_idx)
    return compress_cores(s, cores, factor)


def detect_singularities(disk, mu, r_thr=0.99, gap_deg=10):
    """[保留兼容] 退化面片角度聚类 → 奇异点位置与 α 估计。

    注：α 拟合对拟合半径极敏感（r<0.1→0.34, r<0.2→0.77），且 α≈1 时
    表示非尖角——旧 analytic_extend 因此失败。仅供外部引用，增强不再用它。
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
    """[已废弃，保留兼容] 旧解析延拓实现被证实损坏（fit 病态 → seam 尖刺
    + 角度回绕 → 焊接塌缩/扇形）。勿用；用 sharpen_seam。"""
    raise NotImplementedError(
        "analytic_extend 已被证实损坏（fit 病态 → seam 尖刺 + 角度回绕 → 焊接塌缩）。"
        "请用 sharpen_seam（压缩塌缩核）。"
    )


def angle_resample_boundary(s, n=None):
    """[已废弃，保留兼容] angle_resample 强制 r(θ) 星形 → 毁掉凹形 → 扇形。勿用。"""
    raise NotImplementedError(
        "angle_resample_boundary 被证实毁掉凹形（强制星形 → 扇形失真）。勿用。"
    )
