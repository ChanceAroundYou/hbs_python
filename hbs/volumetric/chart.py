"""球面共形图（boundary chart）与三维 Möbius 变换。

三维固定迹问题里，源域是单位球，目标边界 ``∂Ω`` 先取一张二维共形球面图作为
边界迹；体内映射再在满足这条迹的映照里最小化畸变。这里提供两种图：

- :class:`SpheroidChart`：椭球面（可再复合一个目标侧球 Möbius 变换）；
- :class:`RevolutionChart`：任意光滑旋转曲面，半径剖面由调用方给出。

两者都实现 ``boundary(unit)`` 与 ``cone_initial(points)``，可直接交给
:func:`hbs.volumetric.solve_fixed_trace` 使用。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``scipy.integrate.cumulative_trapezoid(y, x, initial=0)`` 的等价实现。

    该函数到 SciPy 1.6 才加入，而包声明 ``scipy>=1.5``；这里用同一个
    ``cumsum(dx * (y1 + y0) / 2)`` 式子，读数与 SciPy 版本逐位相同。
    """
    return np.concatenate([[0.0], np.cumsum(np.diff(x) * (y[1:] + y[:-1]) / 2.0)])


def ball_mobius(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """保向球自同构 ``M_b``，满足 ``M_b(b) = 0``。

    该有理映射在球外（除极点 ``b/|b|²``）也有定义；逆映射是 ``M_{-b}``。
    """
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)
    bx = x @ b
    x2 = np.sum(x * x, axis=-1)
    b2 = b @ b
    denominator = 1 - 2 * bx + b2 * x2
    return ((1 - b2) * x - (1 - 2 * bx + x2)[..., None] * b) / denominator[..., None]


def mobius_scale(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``M_b`` 在 ``x`` 处的共形伸缩因子 ``(1-|b|²)/|1-2x·b+|b|²|x|²|``。"""
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    return (1 - b @ b) / (1 - 2 * x @ b + (b @ b) * np.sum(x * x, axis=-1))


class SpheroidChart:
    """椭球面 ``x²/a² + z²/c² = 1`` 的解析共形球面图。

    经线映射由一维积分给出：``d ln tan(ψ/2) = (A/r - 1)/sin θ dθ``，
    这里 ``A = sqrt(a²cos²θ + c²sin²θ)`` 是椭球的经线弧长元。可再复合一个
    目标侧球 Möbius 变换 ``target_mobius``；``center=True`` 时按面积重心选
    规范（用 64×128 固定求积，与四面体网格无关）。
    """

    def __init__(
        self,
        a: float = 1.0,
        c: float = 1.5,
        target_mobius: tuple[float, float, float] = (0.0, 0.0, 0.0),
        center: bool = True,
    ):
        self.a, self.c = float(a), float(c)
        self.target_mobius = np.asarray(target_mobius, dtype=float)

        theta = np.linspace(0, np.pi, 65537)
        st, ct = np.sin(theta), np.cos(theta)
        meridian = np.sqrt(self.a**2 * ct**2 + self.c**2 * st**2)
        correction = np.zeros_like(theta)
        correction[1:-1] = (meridian[1:-1] / self.a - 1) / st[1:-1]
        integral = _cumulative_trapezoid(correction, theta)
        integral -= integral[len(theta) // 2]
        psi = np.empty_like(theta)
        psi[0], psi[-1] = 0.0, np.pi
        psi[1:-1] = 2 * np.arctan(np.tan(theta[1:-1] / 2) * np.exp(integral[1:-1]))
        self._theta = PchipInterpolator(psi, theta)

        self.center_parameter = np.zeros(3)
        self.center_success = True
        # 固定的高阶球面求积；不依赖任何四面体网格。
        zz, ww = np.polynomial.legendre.leggauss(64)
        phi = 2 * np.pi * (np.arange(128) + 0.5) / 128
        points = np.stack(
            np.broadcast_arrays(
                np.sqrt(1 - zz[:, None] ** 2) * np.cos(phi)[None, :],
                np.sqrt(1 - zz[:, None] ** 2) * np.sin(phi)[None, :],
                zz[:, None] + np.zeros((1, len(phi))),
            ),
            axis=-1,
        ).reshape(-1, 3)
        _, scales = self.raw_surface(points)
        weights = np.repeat(ww, len(phi)) * scales**2
        weights /= weights.sum()

        def moment(b: np.ndarray) -> np.ndarray:
            return np.sum(weights[:, None] * ball_mobius(points, b), axis=0)

        if center:
            solved = root(moment, np.zeros(3), tol=1e-11)
            self.center_success = bool(
                solved.success or np.linalg.norm(moment(solved.x)) < 1e-10
            )
            if not self.center_success or np.linalg.norm(solved.x) >= 0.95:
                raise RuntimeError("surface area centering failed")
            self.center_parameter = solved.x
        self.area_center_residual = float(np.linalg.norm(moment(self.center_parameter)))
        self._quadrature_source = points
        self._quadrature_weights = weights

    def raw_surface(self, unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把球面点投到椭球面上，同时返回共形伸缩因子。"""
        unit = np.asarray(unit, dtype=float)
        psi = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
        theta = self._theta(psi)
        phi = np.arctan2(unit[..., 1], unit[..., 0])
        st, ct = np.sin(theta), np.cos(theta)
        out = np.stack(
            [self.a * st * np.cos(phi), self.a * st * np.sin(phi), self.c * ct],
            axis=-1,
        )
        # 极点的尺度用邻近极限；求解不使用网格极点。
        safe_psi = np.clip(psi, 1e-7, np.pi - 1e-7)
        scales = self.a * np.sin(self._theta(safe_psi)) / np.sin(safe_psi)
        scales *= mobius_scale(out, self.target_mobius)
        return ball_mobius(out, self.target_mobius), scales

    def boundary(self, unit: np.ndarray) -> np.ndarray:
        """边界迹：先把球面点移到重心规范，再映到椭球面。"""
        uncentered = ball_mobius(unit, -self.center_parameter)
        return self.raw_surface(uncentered)[0]

    def cone_initial(self, points: np.ndarray) -> np.ndarray:
        """锥形初值：沿径向把内部点按对应边界点的半径缩放。

        先把点拉回未移心的球面，取单位方向；对边界点先撤销目标 Möbius，
        乘上半径后再复合回去。原点单独取球面极点，避免除零。
        """
        pre = ball_mobius(np.asarray(points, dtype=float), -self.center_parameter)
        radii = np.linalg.norm(pre, axis=1)
        unit = pre / np.maximum(radii[:, None], 1e-15)
        unit[radii < 1e-14] = [0.0, 0.0, 1.0]
        boundary, _ = self.raw_surface(unit)
        untransformed = ball_mobius(boundary, -self.target_mobius)
        return ball_mobius(radii[:, None] * untransformed, self.target_mobius)

    def audit(self, count: int = 200) -> dict:
        """独立有限差分核验：切空间奇异值比与 Möbius 逆。"""
        rng = np.random.default_rng(20261017)
        unit = rng.normal(size=(count, 3))
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        tangent = rng.normal(size=(count, 3))
        tangent -= np.sum(tangent * unit, axis=1)[:, None] * unit
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        other = np.cross(unit, tangent)
        h = 2e-5
        cols = []
        for direction in (tangent, other):
            plus = np.cos(h) * unit + np.sin(h) * direction
            minus = np.cos(h) * unit - np.sin(h) * direction
            cols.append((self.boundary(plus) - self.boundary(minus)) / (2 * h))
        jac = np.stack(cols, axis=-1)
        singular = np.linalg.svd(jac, compute_uv=False)
        return dict(
            tangent_ratio_max=float(np.max(singular[:, 0] / singular[:, 1])),
            area_center_residual=self.area_center_residual,
            center_parameter=self.center_parameter.tolist(),
            center_success=self.center_success,
            mobius_inverse_error=float(
                np.max(
                    np.abs(
                        ball_mobius(
                            ball_mobius(unit, self.target_mobius), -self.target_mobius
                        )
                        - unit
                    )
                )
            ),
        )


class RevolutionChart:
    """任意光滑旋转曲面的共形球面图。

    与 :class:`SpheroidChart` 用同一套经线积分，只是把椭球的解析经线弧长元
    ``A`` 换成 ``sqrt(r² + r'²)``，其中 ``r(θ)`` 由调用方给出。要求
    ``r > 0`` 且经线映射 ``ψ(θ)`` 严格单调，否则直接报错（曲面不是嵌入的）。
    """

    def __init__(self, radius: Callable[[np.ndarray], np.ndarray], grid: int = 65537):
        self.radius = radius
        theta = np.linspace(0.0, np.pi, grid)
        r = np.asarray(radius(theta), dtype=float)
        if r.shape != theta.shape or not np.all(np.isfinite(r)):
            raise ValueError("radius(theta) must return a finite array of the same shape")
        if np.any(r <= 0):
            raise ValueError("radius(theta) must be strictly positive")

        dr = np.gradient(r, theta)  # 细网格上的中心差分
        arc = np.hypot(r, dr)
        st = np.sin(theta)
        correction = np.zeros_like(theta)
        correction[1:-1] = (arc[1:-1] / r[1:-1] - 1.0) / st[1:-1]
        integral = _cumulative_trapezoid(correction, theta)
        integral -= integral[len(theta) // 2]
        psi = np.empty_like(theta)
        psi[0], psi[-1] = 0.0, np.pi
        psi[1:-1] = 2 * np.arctan(
            np.tan(theta[1:-1] / 2) * np.exp(integral[1:-1])
        )
        if not np.all(np.diff(psi) > 0):
            raise RuntimeError("meridian integral is not strictly increasing")
        self._theta = PchipInterpolator(psi, theta)
        self._grid_theta, self._grid_r = theta, r

    def boundary(self, unit: np.ndarray) -> np.ndarray:
        """球面点 → 旋转曲面点。"""
        unit = np.asarray(unit, dtype=float)
        psi = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
        theta = self._theta(psi)
        r = self.radius(theta)
        phi = np.arctan2(unit[..., 1], unit[..., 0])
        st = np.sin(theta)
        return np.stack(
            [r * st * np.cos(phi), r * st * np.sin(phi), r * np.cos(theta)],
            axis=-1,
        )

    point = boundary

    def cone_initial(self, points: np.ndarray) -> np.ndarray:
        """锥形初值：沿径向把内部点缩放到对应边界点。"""
        points = np.asarray(points, dtype=float)
        radii = np.linalg.norm(points, axis=1)
        unit = points / np.maximum(radii[:, None], 1e-15)
        unit[radii < 1e-14] = [0.0, 0.0, 1.0]
        return radii[:, None] * self.boundary(unit)

    def audit(self, count: int = 200) -> dict:
        """有限差分切向奇异值比，与 :meth:`SpheroidChart.audit` 同一口径。"""
        rng = np.random.default_rng(20261017)
        unit = rng.normal(size=(count, 3))
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        tangent = rng.normal(size=(count, 3))
        tangent -= np.sum(tangent * unit, axis=1)[:, None] * unit
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        other = np.cross(unit, tangent)
        h = 2e-5
        cols = []
        for direction in (tangent, other):
            plus = np.cos(h) * unit + np.sin(h) * direction
            minus = np.cos(h) * unit - np.sin(h) * direction
            cols.append((self.boundary(plus) - self.boundary(minus)) / (2 * h))
        jac = np.stack(cols, axis=-1)
        singular = np.linalg.svd(jac, compute_uv=False)
        return dict(
            tangent_ratio_max=float(np.max(singular[:, 0] / singular[:, 1])),
            tangent_ratio_min=float(np.min(singular[:, 0] / singular[:, 1])),
        )
