"""Boundary-layer realization of the original HBS parent map (project 升级版).

For a smooth, positively oriented, 2*pi-periodic Jordan curve gamma(t),
solve two first-kind logarithmic single-layer equations on the boundary.
The resulting interior and exterior harmonic-measure densities give the
circle phases alpha and beta.  Harmonic extension of
H(exp(i beta(t))) = exp(i alpha(t)) is the original HBS parent map.

The discrete solver uses the periodic logarithm split and Fourier product
integration.  The implementation is for smooth curves, not corners.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root


@dataclass
class DirectHBS:
    boundary: NDArray[np.complex128]
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    rho_internal: NDArray[np.float64]
    rho_external: NDArray[np.float64]
    mobius_center: complex
    internal_equation_residual: float
    external_equation_residual: float
    timings: dict[str, float]
    min_raw_internal_density: float = 0.0
    min_raw_external_density: float = 0.0
    max_raw_internal_density: float = 0.0
    max_raw_external_density: float = 0.0
    density_regularization_mass: float = 0.0
    negative_mass_internal: float = 0.0
    negative_mass_external: float = 0.0
    negative_count_internal: int = 0
    negative_count_external: int = 0
    negative_peak_internal: float = 0.0
    negative_peak_external: float = 0.0

    def density_diagnostics(self) -> dict[str, float | int]:
        """Raw-density negativity snapshot, for acceptance reporting.

        ``negative_mass_*`` is the relative L1 mass of the negative part;
        ``negative_peak_ratio_*`` is the deepest single negative divided by the
        positive peak.  Mass alone cannot tell a discretization artifact from a
        boundary that is outside the engine's domain, so both are reported.
        """
        return {
            "min_raw_internal_density": self.min_raw_internal_density,
            "min_raw_external_density": self.min_raw_external_density,
            "max_raw_internal_density": self.max_raw_internal_density,
            "max_raw_external_density": self.max_raw_external_density,
            "negative_mass_internal": self.negative_mass_internal,
            "negative_mass_external": self.negative_mass_external,
            "negative_count_internal": self.negative_count_internal,
            "negative_count_external": self.negative_count_external,
            "negative_peak_internal": self.negative_peak_internal,
            "negative_peak_external": self.negative_peak_external,
            "negative_peak_ratio_internal": (
                self.negative_peak_internal / self.max_raw_internal_density
                if self.max_raw_internal_density > 0 else float("inf")
            ),
            "negative_peak_ratio_external": (
                self.negative_peak_external / self.max_raw_external_density
                if self.max_raw_external_density > 0 else float("inf")
            ),
            "density_regularization_mass": self.density_regularization_mass,
        }

    def boundary_map(self, theta: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Evaluate H on the source unit circle at arbitrary angles."""
        beta = np.r_[self.beta, 2 * np.pi]
        alpha = np.r_[self.alpha, 2 * np.pi]
        th = np.mod(theta, 2 * np.pi)
        return np.exp(1j * np.interp(th, beta, alpha))

    def interior_map(
        self, z: NDArray[np.complex128], chunk: int = 1000
    ) -> NDArray[np.complex128]:
        """Direct boundary integral, without constructing a welding table."""
        z = np.asarray(z, complex).ravel()
        out = np.empty_like(z)
        x = np.exp(1j * self.beta)
        y = np.exp(1j * self.alpha)
        weight = (2 * np.pi / len(x)) * self.rho_external
        for start in range(0, len(z), chunk):
            zz = z[start : start + chunk]
            kernel = (1 - np.abs(zz[:, None]) ** 2) / np.abs(
                x[None, :] - zz[:, None]
            ) ** 2
            out[start : start + chunk] = kernel @ (y * weight)
        return out

    def moments(self, order: int) -> NDArray[np.complex128]:
        """c_m for m=-order,...,order, with H=sum c_m z^m + c_-m zbar^m."""
        m = np.arange(-order, order + 1)
        phase = np.exp(1j * self.alpha)
        weight = (2 * np.pi / len(phase)) * self.rho_external
        return np.exp(-1j * np.outer(m, self.beta)) @ (phase * weight)

    def analytic_mu(
        self, z: NDArray[np.complex128], order: int = 80
    ) -> NDArray[np.complex128]:
        z = np.asarray(z, complex)
        c = self.moments(order)
        m = np.arange(1, order + 1)
        hz = (
            m[:, None] * c[order + m, None] * z.ravel()[None, :] ** (m[:, None] - 1)
        ).sum(axis=0)
        hc = (
            m[:, None]
            * c[order - m, None]
            * np.conj(z.ravel())[None, :] ** (m[:, None] - 1)
        ).sum(axis=0)
        return (hc / hz).reshape(z.shape)


def _angle_from_density(q: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(q)
    qhat = np.fft.fft(q) / n
    k = np.fft.fftfreq(n, 1 / n)
    antihat = np.zeros(n, complex)
    mask = (k != 0) & (np.abs(k) < n / 2)
    antihat[mask] = 2 * np.pi * qhat[mask] / (1j * k[mask])
    oscillatory = np.fft.ifft(antihat).real * n
    angle = (2 * np.pi * qhat[0].real) * np.arange(n) * (2 * np.pi / n)
    spectral = angle + oscillatory - oscillatory[0]
    if np.min(np.diff(spectral)) > 0 and spectral[-1] < 2 * np.pi:
        return spectral
    # Corner-like traces can have sharply peaked positive densities.  Their
    # truncated Fourier antiderivative may ring and lose monotonicity.
    cumulative = np.r_[0.0, np.cumsum(np.pi * (q[:-1] + q[1:]) * (2 * np.pi / n))]
    return cumulative


def _operator(
    gamma: NDArray[np.complex128], dgamma: NDArray[np.complex128]
) -> NDArray[np.float64]:
    n = len(gamma)
    if n % 2:
        raise ValueError("N must be even")
    dt = 2 * np.pi / n
    k = np.arange(n)
    # Convolution weights for log|2 sin((t-s)/2)| on trigonometric interpolants.
    r = np.zeros(n)
    for m in range(1, n // 2):
        r -= dt * np.cos(m * dt * k) / m
    r -= (2 * np.pi / n**2) * np.cos(np.pi * k)
    delta = np.subtract.outer(np.arange(n), np.arange(n))
    angle = dt * delta
    denominator = 2 * np.abs(np.sin(angle / 2))
    distance = np.abs(gamma[:, None] - gamma[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        smooth = np.log(distance / denominator)
    np.fill_diagonal(smooth, np.log(np.abs(dgamma)))
    return r[np.mod(delta, n)] + dt * smooth


def _density_tolerances_are_exceeded(
    min_raw: float,
    max_raw: float,
    negative_mass: float,
    negative_peak: float,
    negative_mass_tolerance: float,
    negative_peak_tolerance: float,
) -> bool:
    """Single place where the negative-density acceptance contract is decided.

    Roundoff-level negatives (``min(q) >= -1e-10 * max(q)``) are always allowed and
    get clipped.  Anything deeper has to satisfy **both** bounds: the relative L1
    mass *and* the deepest single negative relative to the positive peak.  A cusp or
    a heavily noisy contour keeps a small total mass while its pointwise negatives
    stay of the same order as the density, so the mass bound alone cannot tell a
    discretization artifact from a boundary that is outside the engine's domain.
    """
    if min_raw >= -1e-10 * max_raw:
        return False
    peak_ratio = negative_peak / max_raw if max_raw > 0 else float("inf")
    return negative_mass > negative_mass_tolerance or peak_ratio > negative_peak_tolerance


def compute_direct_hbs(
    curve: Callable[[NDArray[np.float64]], NDArray[np.complex128]],
    derivative: Callable[[NDArray[np.float64]], NDArray[np.complex128]],
    n: int = 480,
    pole: complex = 0j,
    negative_mass_tolerance: float = 0.0,
    negative_peak_tolerance: float = 0.0,
) -> DirectHBS:
    """Solve for the circle phases on the ``n``-point collocation grid.

    ``curve`` and ``derivative`` must parameterize a smooth CCW boundary on
    [0,2*pi), with ``pole`` strictly inside.  Both logarithmic single-layer
    equations are collocated at the ``n`` uniform nodes, which is also the grid
    the M1 centering sum runs over.

    Negatives in the raw density are tolerated only when they look like
    roundoff: ``min(q) >= -1e-10 * max(q)``.  Beyond that both tolerances must
    hold, per side:

    * ``negative_mass_tolerance`` bounds the relative L1 mass
      ``sum(|min(q,0)|) / sum(|q|)``;
    * ``negative_peak_tolerance`` bounds the *deepest single* negative relative
      to the positive peak, ``max(|min(q,0)|) / max(q)``.

    The second bound is what separates a discretization artifact from a boundary
    that is outside the engine's domain: a cusp or a heavily noisy contour keeps
    negatives of the same order as the density itself, and a small total mass
    does not make those pointwise negatives geometrically meaningful.  The
    defaults ``0``/``0`` keep the strict contract (only roundoff-level negatives
    are clipped).  Callers that deliberately accept discretization-level
    negatives pass small positive values and must report the measured mass and
    peak ratio.
    """
    if n % 2:
        raise ValueError("N must be even")
    tic = perf_counter()
    t = 2 * np.pi * np.arange(n) / n
    gamma = np.asarray(curve(t), complex)
    gp = np.asarray(derivative(t), complex)
    if gamma.shape != (n,) or gp.shape != (n,) or np.min(np.abs(gp)) <= 0:
        raise ValueError("curve and derivative must be regular length-N arrays")
    matrix = _operator(gamma, gp)
    system = np.empty((n + 1, n + 1), float)
    system[:n, :n] = matrix
    system[:n, n] = 1
    system[n, :n] = 2 * np.pi / n
    system[n, n] = 0
    rhs = np.zeros((n + 1, 2))
    rhs[:n, 0] = np.log(np.abs(gamma - pole))
    rhs[n, :] = 1
    t_matrix = perf_counter()
    lu, piv = lu_factor(system)
    sol = lu_solve((lu, piv), rhs)
    q_in, q_out = sol[:n, 0], sol[:n, 1]
    min_raw_in = float(q_in.min())
    min_raw_out = float(q_out.min())
    r_in = float(np.max(np.abs(system @ sol[:, 0] - rhs[:, 0])))
    r_out = float(np.max(np.abs(system @ sol[:, 1] - rhs[:, 1])))
    t_solve = perf_counter()
    neg_mass_in = float(np.abs(np.minimum(q_in, 0.0)).sum() / np.abs(q_in).sum())
    neg_mass_out = float(np.abs(np.minimum(q_out, 0.0)).sum() / np.abs(q_out).sum())
    neg_count_in = int(np.count_nonzero(q_in < 0))
    neg_count_out = int(np.count_nonzero(q_out < 0))
    neg_peak_in = float(np.abs(np.minimum(q_in, 0.0)).max())
    neg_peak_out = float(np.abs(np.minimum(q_out, 0.0)).max())
    max_raw_in = float(q_in.max())
    max_raw_out = float(q_out.max())
    correction_mass = 0.0
    for q, max_raw, mass, peak in ((q_in, max_raw_in, neg_mass_in, neg_peak_in),
                                   (q_out, max_raw_out, neg_mass_out, neg_peak_out)):
        if _density_tolerances_are_exceeded(
            q.min(), max_raw, mass, peak,
            negative_mass_tolerance, negative_peak_tolerance,
        ):
            raise RuntimeError(
                "negative harmonic-measure density beyond tolerance "
                f"(mass {mass:.3e} > {negative_mass_tolerance:.1e}, peak ratio "
                f"{peak / max_raw:.3e} > {negative_peak_tolerance:.1e}); "
                "refine or check boundary"
            )
        if q.min() <= 0:
            old = q.copy()
            np.maximum(q, 1e-12 * q.max(), out=q)
            q /= np.sum(q) * (2 * np.pi / n)
            correction_mass += float(np.sum(np.abs(q - old)) * (2 * np.pi / n))
    alpha0 = _angle_from_density(q_in)
    beta = _angle_from_density(q_out)
    if np.min(np.diff(beta)) <= 0 or beta[-1] >= 2 * np.pi:
        raise RuntimeError("exterior angle is not monotone; refine discretization")

    # M1 规范化（论文式 5.11 / Algorithm 6.2）：Σ_k Φ1^{-1}(z_k) = 0。
    #
    # 求和必须覆盖**全部 N 个离散点**，不是输入样本的子集。早期实现按输入点数
    # （通常 500）取子集，而 α 定义在 n=1024 网格上，两者不一致：全网格上的
    # 残差实测 2.6e-3 ~ 2.8e-2，`|a|` 也随之偏大。论文 Algorithm 6.2 明确写的是
    # "p_i, i = 1,…,N" 且 N 与求解网格一致。
    z_full = np.exp(1j * alpha0)

    def center_equations(v: NDArray[np.float64]) -> NDArray[np.float64]:
        a = complex(v[0], v[1])
        mapped = (z_full - a) / (1 - np.conj(a) * z_full)
        mean = mapped.mean()
        return np.array([mean.real, mean.imag])

    result = root(center_equations, np.array([z_full.mean().real, z_full.mean().imag]))
    a = complex(*result.x)
    if abs(a) >= 1 or np.linalg.norm(center_equations(result.x)) > 1e-9:
        raise RuntimeError("inner Möbius centering failed")
    # On the unit circle, arg((z-a)/(1-conj(a)z)) equals
    # arg(z)+2 arg(1-a/z).  Since |a|<1, the second
    # argument is continuous on the principal branch.  This avoids a
    # false unwrap jump when one discretization cell carries >pi angle.
    alpha = alpha0 + 2 * np.angle(1 - a * np.exp(-1j * alpha0))
    alpha -= alpha[0]
    if np.min(np.diff(alpha)) <= 0:
        raise RuntimeError("inner angle is not monotone")
    done = perf_counter()
    return DirectHBS(
        gamma,
        alpha,
        beta,
        q_in,
        q_out,
        a,
        r_in,
        r_out,
        {
            "matrix": t_matrix - tic,
            "solve": t_solve - t_matrix,
            "gauge": done - t_solve,
            "total": done - tic,
        },
        min_raw_internal_density=min_raw_in,
        min_raw_external_density=min_raw_out,
        max_raw_internal_density=max_raw_in,
        max_raw_external_density=max_raw_out,
        density_regularization_mass=correction_mass,
        negative_mass_internal=neg_mass_in,
        negative_mass_external=neg_mass_out,
        negative_count_internal=neg_count_in,
        negative_count_external=neg_count_out,
        negative_peak_internal=neg_peak_in,
        negative_peak_external=neg_peak_out,
    )


def compute_direct_hbs_from_boundary(
    bound: NDArray[np.float64], n: int = 480
) -> DirectHBS:
    """Point-array API matching ``hbs.get_hbs`` for smooth Jordan curves.

    Fits a periodic cubic spline in normalized arclength.  Its derivative is
    supplied to the logarithmic-kernel diagonal.  Corners and rough contours
    require a different quadrature and are outside this prototype's scope.
    """
    bound = np.asarray(bound, float)
    if bound.ndim != 2 or bound.shape[1] != 2 or len(bound) < 8:
        raise ValueError("bound must be an n-by-2 smooth closed curve sample")
    z = bound[:, 0] + 1j * bound[:, 1]
    if np.abs(z[0] - z[-1]) < 1e-13:
        z = z[:-1]
    area = np.sum((z.real * np.roll(z.imag, -1)) - (z.imag * np.roll(z.real, -1)))
    if area < 0:
        z = z[::-1]
    edges = np.abs(np.diff(np.r_[z, z[0]]))
    if edges.min() <= 1e-14:
        raise ValueError("repeated or degenerate boundary points")
    s = np.r_[0.0, np.cumsum(edges)]
    t = 2 * np.pi * s / s[-1]
    x = CubicSpline(t, np.r_[z.real, z.real[0]], bc_type="periodic")
    y = CubicSpline(t, np.r_[z.imag, z.imag[0]], bc_type="periodic")
    curve = lambda v: x(v) + 1j * y(v)
    derivative = lambda v: x(v, 1) + 1j * y(v, 1)
    pole = _interior_pole(z)
    return compute_direct_hbs(curve, derivative, n=n, pole=pole)


def _inside_polygon(point: complex, vertices: NDArray[np.complex128]) -> bool:
    """Odd-even test for the polygon through the sampled boundary points."""
    x, y = point.real, point.imag
    x0, y0 = vertices.real, vertices.imag
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    cross = (y0 > y) != (y1 > y)
    xhit = x0 + (y - y0) * (x1 - x0) / np.where(y1 == y0, 1.0, y1 - y0)
    return bool(np.count_nonzero(cross & (x < xhit)) % 2)


def _interior_pole(vertices: NDArray[np.complex128]) -> complex:
    """Choose a verified interior Green pole, including for concave curves."""
    mean = complex(np.mean(vertices))
    if _inside_polygon(mean, vertices):
        return mean
    xlo, xhi = float(vertices.real.min()), float(vertices.real.max())
    ylo, yhi = float(vertices.imag.min()), float(vertices.imag.max())
    xs = np.linspace(xlo, xhi, 33)[1:-1]
    ys = np.linspace(ylo, yhi, 33)[1:-1]
    candidates = [
        complex(x, y)
        for x in xs
        for y in ys
        if _inside_polygon(complex(x, y), vertices)
    ]
    if not candidates:
        raise ValueError("could not locate an interior Green pole")
    starts = vertices
    edges = np.roll(vertices, -1) - vertices

    def clearance(p: complex) -> float:
        t = np.clip(np.real((p - starts) * np.conj(edges)) / np.abs(edges) ** 2, 0, 1)
        return float(np.min(np.abs(p - (starts + t * edges))))

    return max(candidates, key=clearance)
