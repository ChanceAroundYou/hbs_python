import numpy as np


def zipper(bound, others=None):
    """zipper 共形焊接（边界 → 单位圆）。

    已知局限：高对称形状（如 5 角星）中途大量点落虚轴/原点，微旋防护
    虽消除 ValueError 崩溃，但累积旋转会把输出压到实线（退化）→ 后续
    y_post_norm 不收敛。这是 zipper 对对称形状的结构性退化，非本库可简单修复。
    非对称形状（鸟/椭圆/多边形）正常。
    """
    if others is None:
        others = []

    # 纯虚数稳健化：边界含实部≈0 的点会触发 f() 的纯虚数检查（数值误崩）。
    # 绕质心微旋 0.5° 避开——HBS 归一化吸收输入旋转，输出代表不变。
    all_pts = np.concatenate([bound, others])
    if np.any(np.abs(all_pts.real) < 1e-9):
        c = np.mean(all_pts)
        all_pts = (all_pts - c) * np.exp(1j * np.deg2rad(0.5)) + c
        # 质心在虚轴上时绕质心旋转不动点仍在虚轴 → 加微小平移（HBS 平移不变）
        if np.any(np.abs(all_pts.real) < 1e-12):
            all_pts = all_pts + 1e-6
        bound, others = all_pts[: len(bound)], all_pts[len(bound) :]

    n = len(bound)
    params = np.zeros(n + 1, dtype=complex)
    params[:2] = bound[:2]
    points = f_pre(np.concatenate([bound, others]), params[0], params[1])
    for j in range(2, n):
        params[j] = points[j]
        points = f(points, points[j])
    params[n] = points[0]
    points = f_end(points, params[n])
    points = f_final(points)

    bound = points[:n]
    others = points[n:]
    return bound, others, params



def f_pre(z, p, q):
    """
    map z to right half plane, where real part is positive.
    p -> inf,
    q -> 0,
    note that always p = z[0], q = z[1]
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (z - q) / (z - p)
    w[np.isinf(z)] = 1
    w[z == p] = np.inf
    w = np.sqrt(w)
    return w


def f1(z, p):
    """
    0 -> 0
    p -> 1
    aj -> bj
    inf -> - c/d * 1j
    1j/d -> inf
    """
    c = np.real(p) / np.abs(p) ** 2
    d = np.imag(p) / np.abs(p) ** 2
    # if d == 0:
    #     s = z[:10]
    #     print(s, p)
    #     print(s.imag)
    #     from matplotlib import pyplot as plt
    #     plt.scatter(s.real, s.imag)
    #     plt.show()
    if d == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = z * c
        w[np.isinf(z)] = np.inf
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = c * z / (1  + 1j * d * z)
        w[np.isinf(z)] = -c / d * 1j
        w[z == 1j / d] = np.inf
        w[np.isclose(z, p)] = 1
    return w


def f2(z):
    """
    1 -> 0
    0 -> 1j
    bj -> sqrt(b^2 + 1)j (positive imaginary part)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.sqrt(z**2 - 1)
    k = np.imag(w) * np.imag(z) < 0
    w[k] = -w[k]
    w[z == 0] = 1j
    w[np.isinf(z)] = np.inf
    return w


def f(z, p):
    """
    map p to 0,
    with keep points in y-axis, going up.
    0 -> 0 -> 1j
    p -> 1 -> 0
    aj -> bj -> sqrt(b^2 + 1)j
    """
    if abs(p) < 1e-9 or (abs(p.imag) > 1e-9 and abs(p.real) < 1e-12):
        # 退化稳健化：仅处理真正会让 f1 爆掉的情形——
        #  (a) p≈原点（c=real/|p|²→NaN），旋转后仍贴原点则加微移；
        #  (b) p 恰好在虚轴（real==0，原代码 raise）——微旋 0.5° 避开。
        # 注意：近轴但非退化（|real|≈1e-6）不触发，避免破坏正常 seam。
        # 微旋/微移对共形映射可忽略（归一化吸收）。
        rot = np.exp(1j * np.deg2rad(0.5))
        z = z * rot
        p = p * rot
        if abs(p) < 1e-12:
            z = z + 1e-6
            p = p + 1e-6

    w = f1(z, p)
    w = f2(w)
    w[np.isclose(z, p)] = 0
    return w


def f_end(z, p):
    if abs(p) < 1e-9:
        # 仅处理 p≈原点（q=1-z/p 除零）。恰在虚轴/实轴对 z/p 无害，
        # 不触发——避免破坏正常 seam（bird 等 points[0] 常恰在虚轴）。
        rot = np.exp(1j * np.deg2rad(0.5))
        z = z * rot
        p = p * rot
        if abs(p) < 1e-12:
            z = z + 1e-6
            p = p + 1e-6
    q = 1 - z / p
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (z / q) ** 2
    w[np.isinf(z)] = p**2
    w[q == 0] = np.inf
    return w


def f_final(z):
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (z - 1j) / (z + 1j)
    w[np.isinf(z)] = 1
    w[z == -1j] = np.inf
    return w


def zipper_params(points, params):
    n = len(params) - 1
    points = f_pre(points, params[0], params[1])
    for j in range(2, n):
        points = f(points, params[j])
    points = f_end(points, params[n])
    points = f_final(points)
    return points


def zipper_inv(points, params):
    n = len(params) - 1
    points = f_final_inv(points)
    points = f_end_inv(points, params[n])
    for j in range(n - 1, 1, -1):
        points = f_inv(points, params[j])
    points = f_pre_inv(points, params[0], params[1])
    return points


def f_pre_inv(w, p, q):
    z = (p * w**2 - q) / (w**2 - 1)
    z[np.isinf(w)] = p
    z[w == 1] = np.inf
    return z


def f1_inv(w, p):
    pc = np.real(p) / np.abs(p) ** 2
    pd = np.imag(p) / np.abs(p) ** 2
    z = w / (pc - 1j * pd * w)
    z[np.isinf(w)] = 1j / pd
    z[w == -1j * pc / pd] = np.inf
    return z


def f2_inv(w):
    z = np.sqrt(w**2 + 1)
    k = np.imag(w) * np.imag(z) < 0
    z[k] = -z[k]
    z[w == 1j] = 0
    z[np.isinf(w)] = np.inf
    return z


def f_inv(w, p):
    z = f2_inv(w)
    z = f1_inv(z, p)
    return z


def f_end_inv(w, p):
    z = np.sqrt(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = z / (1 + z / p)
    z[np.isinf(w)] = p
    return z


def f_final_inv(w):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (w + 1) * 1j / (1 - w)
    k = np.abs(np.imag(z)) < 1e-10
    z[k] = np.real(z[k])
    z[np.isinf(w)] = -1j
    z[w == 1] = np.inf
    return z
