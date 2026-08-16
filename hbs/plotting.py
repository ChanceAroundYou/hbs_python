"""可选绘图工具——核心数学模块不依赖 matplotlib。

用法：
    from hbs.plotting import plot_mesh, plot_mu, plot_mapping, plot_boundary, \
        plot_welding, plot_conformal
    plot_mesh(mesh)
    plot_mu(mesh, mu)
    plot_mapping(mesh, mapping)
    plot_boundary(boundary)
    plot_conformal(cw)
    plot_welding(cw, is_interp=True)

matplotlib 仅在调用绘图时惰性 import，因此不加 matplotlib 也能用全部
计算功能（解耦后核心包不再把 matplotlib 列为硬依赖）。
"""
import numpy as np


def plot_mesh(mesh, with_face_center=False):
    """绘制三角网格。"""
    import matplotlib.pyplot as plt

    plt.gca().set_aspect("equal", adjustable="box")
    plt.triplot(mesh.vert[:, 0], mesh.vert[:, 1], mesh.face)
    if with_face_center:
        plt.plot(mesh.face_center[:, 0], mesh.face_center[:, 1], "o")
    plt.show()


def plot_mu(mesh, mu, is_3d=False):
    """按面心绘制 |μ| 标量场。"""
    import matplotlib.pyplot as plt

    assert isinstance(mu, np.ndarray) and np.issubdtype(
        mu.dtype, np.complexfloating
    ), "mu must be complex array"
    assert mu.ndim == 1 and mu.shape[0] == mesh.face_num, (
        "mu must be 1D array with length equal to face number"
    )

    magnitude = np.abs(mu)
    magnitude[magnitude > 1] = 1
    if not is_3d:
        plt.scatter(
            mesh.face_center[:, 0],
            mesh.face_center[:, 1],
            c=magnitude,
            cmap="jet",
            s=0.5,
        )
        plt.colorbar(label="Magnitude")
    else:
        ax = plt.axes(projection="3d")
        ax.scatter(
            mesh.face_center[:, 0],
            mesh.face_center[:, 1],
            magnitude,
            c=magnitude,
            cmap="jet",
            s=0.5,
        )
    plt.show()


def plot_mapping(mesh, mapping, is_3d=False):
    """按顶点绘制映射坐标模长场。"""
    import matplotlib.pyplot as plt

    assert isinstance(mapping, np.ndarray) and np.issubdtype(
        mapping.dtype, np.floating
    ), "mapping must be real float array"
    assert mapping.ndim == 2 and mapping.shape == (mesh.vert_num, 2), (
        "mapping must be n x 2 array"
    )

    magnitude = np.linalg.norm(mapping, axis=1)
    if not is_3d:
        plt.scatter(
            mesh.vert[:, 0],
            mesh.vert[:, 1],
            c=magnitude,
            cmap="jet",
            s=0.5,
        )
        plt.colorbar(label="Magnitude")
    else:
        ax = plt.axes(projection="3d")
        ax.scatter(
            mesh.vert[:, 0],
            mesh.vert[:, 1],
            magnitude,
            c=magnitude,
            cmap="jet",
            s=0.5,
        )
    plt.show()


def plot_boundary(boundary, with_line=True, scale=1):
    """绘制边界点（线 + 散点）。"""
    import matplotlib.pyplot as plt

    num_points = boundary.shape[0]
    x_min, x_max = np.min(boundary[:, 0]), np.max(boundary[:, 0])
    y_min, y_max = np.min(boundary[:, 1]), np.max(boundary[:, 1])
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    fig_size = (num_points / 60 * scale * aspect_ratio, num_points / 60 * scale)
    plt.figure(figsize=fig_size)
    if with_line:
        plt.plot(
            boundary[:, 0],
            boundary[:, 1],
            linestyle="-",
            linewidth=1 * scale,
            color="white",
            label="Boundary Line",
        )
    plt.scatter(
        boundary[:, 0],
        boundary[:, 1],
        marker="o",
        s=15 * scale,
        edgecolor="lightgreen",
        facecolor="none",
        label="Boundary Points",
    )
    plt.show()


def plot_conformal(cw):
    """叠加绘制 x 与 y（焊接的边界/内部点）。"""
    import matplotlib.pyplot as plt

    plt.gca().set_aspect("equal", adjustable="box")
    plt.scatter(cw.x.real, cw.x.imag, c="C0", label="x")
    plt.scatter(cw.y.real, cw.y.imag, c="C1", label="y")
    plt.legend()
    plt.show()


def plot_welding(cw, is_interp=True):
    """绘制焊接映射的角度曲线（x_angle → y_angle）。"""
    import matplotlib.pyplot as plt

    plt.gca().set_aspect("equal", adjustable="box")
    x_angle = np.angle(cw.x)
    x_angle = (x_angle - x_angle[0]) % (2 * np.pi)
    y_angle = cw.get_y_angle()

    if is_interp:
        plt.plot(x_angle, y_angle, linestyle="-", linewidth=2)
    else:
        plt.scatter(x_angle, y_angle, s=2)
    plt.show()
