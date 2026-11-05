"""Riemann maps from interior harmonic measure / the Green function.

For a simply connected Jordan domain ``Omega`` and an interior Green pole
``z0``, let ``G(z) = G_Omega(z, z0)`` and ``H`` be its harmonic conjugate.
Then

    Phi(z) = G(z) + i H(z),       Psi(z) = exp(-Phi(z))

is holomorphic, maps ``Omega`` onto the unit disk, and its inverse is a
Riemann map.  On the boundary, ``G = 0`` and

    dH(gamma(t))/dt = dG(gamma(t))/dn |gamma'(t)| = q(t),

where ``q`` is the interior harmonic-measure density.  Integrating ``q``
gives the circle phase ``alpha(t)`` and hence the boundary correspondence

    F(exp(i alpha(t))) = gamma(t).

This module computes ``alpha`` with the existing harmonic-measure solver and
evaluates ``F`` by the Cauchy integral

    F(w) = (1 / (2*pi*i)) int_{|zeta|=1} f(zeta)/(zeta-w) dzeta.

It does **not** use harmonic extension, and it does not identify the HBS
parent map ``H: D -> D`` with the Riemann map ``F: D -> Omega``.

Normalization
-------------
``F(0)`` is whatever the harmonic-measure M1 normalization selects; it is
reported in ``diagnostics.f_at_zero`` and is not assumed to be zero.  A
source-circle rotation is applied so that ``F'(0)`` is positive real:

    F_normalized(w) = F(exp(i*phi) w),   phi = -arg(F'(0)).

The target domain is unchanged.  The returned boundary is sampled at uniform
source angles after this rotation.

Scope
-----
The solver requires a smooth positively oriented Jordan boundary with a
strictly positive interior harmonic-measure density.  Corners, cusps, rough
pixel contours and multiply connected domains are outside this prototype;
failures are explicit rather than silently falling back.

The finite-difference Cauchy--Riemann residual in
:meth:`RiemannMap.cauchy_riemann_residual` is a numerical check, not a
continuity proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from hbs._harmonic_measure import direct
from hbs.harmonic_measure import _SplineShape

__all__ = ["RiemannMap", "RiemannMapDiagnostics", "riemann_map"]


@dataclass(frozen=True)
class RiemannMapDiagnostics:
    """Diagnostics from the finite harmonic-measure solve."""

    n_boundary: int
    min_phase_step: float
    min_raw_internal_density: float
    f_at_zero: complex
    f_prime_at_zero: complex
    source_rotation: float
    min_abs_derivative: float
    max_abs_beltrami: float
    cr_grid_h: float

    def as_dict(self) -> dict:
        return asdict(self)


def _as_complex_points(values, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 2:
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} must be an (n, 2) floating array")
        points = array[:, 0] + 1j * array[:, 1]
    elif array.ndim == 1 and np.issubdtype(array.dtype, np.complexfloating):
        points = array.astype(complex)
    else:
        raise ValueError(f"{name} must be an (n, 2) real array or a complex vector")
    if points.size < 4 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain at least four finite points")
    return points


def _signed_area(points: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            np.imag(np.conj(points) * np.roll(points, -1))
        )
    )


def _cauchy_eval(points, boundary_values, circle_points, *, chunk: int = 4096):
    """Evaluate the holomorphic Cauchy integral of unit-circle boundary data."""
    points = np.asarray(points, complex).ravel()
    values = np.asarray(boundary_values, complex)
    zeta = np.asarray(circle_points, complex)
    if values.shape != zeta.shape:
        raise ValueError("boundary_values and circle_points must have the same shape")
    dtheta = 2.0 * np.pi / zeta.size
    out = np.empty(points.shape, complex)
    for start in range(0, points.size, chunk):
        block = points[start:start + chunk]
        out[start:start + chunk] = (dtheta / (2.0 * np.pi)) * (
            (values * zeta)[None, :] / (zeta[None, :] - block[:, None])
        ).sum(axis=1)
    return out


def _cauchy_derivative(points, boundary_values, circle_points, *, chunk: int = 4096):
    """Evaluate the derivative of the Cauchy integral."""
    points = np.asarray(points, complex).ravel()
    values = np.asarray(boundary_values, complex)
    zeta = np.asarray(circle_points, complex)
    if values.shape != zeta.shape:
        raise ValueError("boundary_values and circle_points must have the same shape")
    dtheta = 2.0 * np.pi / zeta.size
    out = np.empty(points.shape, complex)
    for start in range(0, points.size, chunk):
        block = points[start:start + chunk]
        out[start:start + chunk] = (dtheta / (2.0 * np.pi)) * (
            (values * zeta)[None, :] / (zeta[None, :] - block[:, None]) ** 2
        ).sum(axis=1)
    return out


def _five_point_cr(
    evaluator, *, grid: int = 81, radius: float = 0.6
) -> dict:
    """Five-point Cauchy--Riemann residual on a square grid in the disk."""
    grid = int(grid)
    radius = float(radius)
    if grid < 7:
        raise ValueError("grid must be at least 7")
    if not (0.0 < radius < 0.9):
        raise ValueError("radius must satisfy 0 < radius < 0.9")
    h = 2.0 * radius / (grid - 1)
    x = np.linspace(-radius, radius, grid)
    X, Y = np.meshgrid(x, x)
    W = X + 1j * Y
    F = np.asarray(evaluator(W), complex)
    Fx = np.full_like(F, np.nan + 0j)
    Fy = np.full_like(F, np.nan + 0j)
    Fx[:, 2:-2] = (-F[:, 4:] + 8.0 * F[:, 3:-1] - 8.0 * F[:, 1:-3] + F[:, :-4]) / (12.0 * h)
    Fy[2:-2, :] = (-F[4:, :] + 8.0 * F[3:-1, :] - 8.0 * F[1:-3, :] + F[:-4, :]) / (12.0 * h)
    Fz = 0.5 * (Fx - 1j * Fy)
    Fzbar = 0.5 * (Fx + 1j * Fy)
    mask = np.hypot(X, Y) <= 0.9 * radius
    if not np.any(mask):
        raise ValueError("grid has no interior points")
    safe = np.abs(Fz) > 1e-14
    mu = np.divide(Fzbar, Fz, out=np.zeros_like(Fz), where=safe)
    return {
        "grid": grid,
        "radius": radius,
        "h": h,
        "max_abs_Fzbar": float(np.nanmax(np.abs(Fzbar[mask]))),
        "max_abs_mu": float(np.nanmax(np.abs(mu[mask]))),
        "min_abs_Fz": float(np.nanmin(np.abs(Fz[mask]))),
    }


class RiemannMap:
    """A numerically computed Riemann map ``D -> Omega``.

    The constructor is private; use :meth:`from_boundary` or
    :meth:`from_curve`.
    """

    def __init__(
        self,
        curve,
        alpha: np.ndarray,
        alpha_inverse: PchipInterpolator,
        boundary_values: np.ndarray,
        theta: np.ndarray,
        t: np.ndarray,
        source_rotation: float,
        diagnostics: RiemannMapDiagnostics,
    ):
        self._curve = curve
        self.alpha = np.asarray(alpha, float)
        self._alpha_inverse = alpha_inverse
        self._boundary_values = np.asarray(boundary_values, complex)
        self._theta = np.asarray(theta, float)
        self._zeta = np.exp(1j * self._theta)
        self.t = np.asarray(t, float)
        self.source_rotation = float(source_rotation)
        self.diagnostics = diagnostics

    @classmethod
    def from_boundary(cls, bound, *, n: int = 1024, diagnostics: bool = True):
        """Build a Riemann map from ``n x 2`` boundary samples.

        The boundary is converted to CCW and fitted with the same periodic
        spline used by the harmonic-measure engine.  Use ``from_curve`` when
        an analytic parameterization and derivative are available.
        """
        points = _as_complex_points(bound, name="bound")
        shape = _SplineShape(np.column_stack([points.real, points.imag]))
        return cls._build(
            shape.curve, shape.derivative, n=n, diagnostics=diagnostics
        )

    @classmethod
    def from_curve(cls, curve, derivative, *, n: int = 1024, diagnostics: bool = True):
        """Build a Riemann map from vectorized ``curve(t)`` / ``derivative(t)``.

        Inputs must be CCW; clockwise parameterizations are reversed.  Both
        callables must accept a 1-D real array and return a complex array of
        the same shape.
        """
        if not callable(curve) or not callable(derivative):
            raise TypeError("curve and derivative must be callable")

        probe_t = 2.0 * np.pi * np.arange(64) / 64
        probe = np.asarray(curve(probe_t), complex)
        if probe.shape != probe_t.shape or not np.all(np.isfinite(probe)):
            raise ValueError("curve must return one finite complex value per input")
        if _signed_area(probe) < 0.0:
            original_curve = curve
            original_derivative = derivative

            def curve(s):
                return original_curve(2.0 * np.pi - np.asarray(s, float))

            def derivative(s):
                return -original_derivative(2.0 * np.pi - np.asarray(s, float))

        return cls._build(curve, derivative, n=n, diagnostics=diagnostics)

    @classmethod
    def _build(cls, curve, derivative, *, n: int, diagnostics: bool):
        n = int(n)
        if n < 64 or n % 2:
            raise ValueError("n must be an even integer >= 64")
        t_grid = 2.0 * np.pi * np.arange(n) / n
        points = np.asarray(curve(t_grid), complex)
        if points.shape != t_grid.shape or not np.all(np.isfinite(points)):
            raise ValueError("curve returned an invalid boundary sample")
        pole = direct._interior_pole(points)
        result = direct.compute_direct_hbs(
            curve, derivative, n=n, pole=pole
        )
        alpha = np.asarray(result.alpha, float)
        if alpha.shape != (n,) or not np.all(np.isfinite(alpha)):
            raise RuntimeError("harmonic-measure solver returned an invalid phase")
        if np.min(np.diff(alpha)) <= 0:
            raise RuntimeError("interior phase is not strictly monotone")

        alpha = alpha - alpha[0]
        alpha_ext = np.r_[alpha, 2.0 * np.pi]
        t_ext = np.r_[t_grid, 2.0 * np.pi]
        alpha_inverse = PchipInterpolator(alpha_ext, t_ext)
        theta = 2.0 * np.pi * np.arange(n) / n
        t = alpha_inverse(theta)
        boundary_values = np.asarray(curve(t), complex)
        zeta = np.exp(1j * theta)

        derivative_at_zero = _cauchy_derivative(
            np.array([0j]), boundary_values, zeta
        )[0]
        if not np.isfinite(derivative_at_zero) or abs(derivative_at_zero) < 1e-14:
            raise RuntimeError("Riemann map has a degenerate derivative at the source origin")
        rotation = -float(np.angle(derivative_at_zero))

        if diagnostics:
            cr = _five_point_cr(
                lambda w: _cauchy_eval(
                    np.asarray(w, complex).ravel() * np.exp(1j * rotation),
                    boundary_values,
                    zeta,
                ).reshape(np.asarray(w).shape),
                grid=81,
                radius=0.6,
            )
            min_abs_derivative = cr["min_abs_Fz"]
            max_abs_beltrami = cr["max_abs_mu"]
            cr_grid_h = cr["h"]
        else:
            # Still evaluate the derivative at the origin; the CR grid is opt-in.
            min_abs_derivative = float(abs(derivative_at_zero))
            max_abs_beltrami = float("nan")
            cr_grid_h = float("nan")

        f_at_zero = _cauchy_eval(np.array([0j]), boundary_values, zeta)[0]
        f_prime_at_zero = derivative_at_zero * np.exp(1j * rotation)
        profile = RiemannMapDiagnostics(
            n_boundary=n,
            min_phase_step=float(np.min(np.diff(alpha))),
            min_raw_internal_density=float(result.min_raw_internal_density),
            f_at_zero=complex(f_at_zero),
            f_prime_at_zero=complex(f_prime_at_zero),
            source_rotation=rotation,
            min_abs_derivative=float(min_abs_derivative),
            max_abs_beltrami=float(max_abs_beltrami),
            cr_grid_h=float(cr_grid_h),
        )
        return cls(
            curve=curve,
            alpha=alpha,
            alpha_inverse=alpha_inverse,
            boundary_values=boundary_values,
            theta=theta,
            t=t,
            source_rotation=rotation,
            diagnostics=profile,
        )

    def _as_output(self, values: np.ndarray, shape: tuple[int, ...]):
        if shape == ():
            return complex(values[0])
        return values.reshape(shape)

    def __call__(self, points):
        """Evaluate ``F`` at complex points in the unit disk."""
        array = np.asarray(points, complex)
        flat = array.ravel()
        shifted = flat * np.exp(1j * self.source_rotation)
        values = _cauchy_eval(shifted, self._boundary_values, self._zeta)
        return self._as_output(values, array.shape)

    def derivative(self, points):
        """Evaluate ``F'(z)`` at complex points in the unit disk."""
        array = np.asarray(points, complex)
        flat = array.ravel()
        shifted = flat * np.exp(1j * self.source_rotation)
        values = np.exp(1j * self.source_rotation) * _cauchy_derivative(
            shifted, self._boundary_values, self._zeta
        )
        return self._as_output(values, array.shape)

    def cauchy_riemann_residual(self, *, grid: int = 81, radius: float = 0.6) -> dict:
        """Numerical five-point ``F_zbar`` / ``max |mu|`` check.

        This is a finite-difference diagnostic on a grid in the source disk;
        it is not a proof of conformality.
        """
        return _five_point_cr(self, grid=grid, radius=radius)

    @property
    def boundary(self) -> np.ndarray:
        """Normalized boundary values ``F(exp(i theta))`` at uniform source angles."""
        shifted = np.mod(self._theta + self.source_rotation, 2.0 * np.pi)
        return np.asarray(self._curve(self._alpha_inverse(shifted)), complex)

    @property
    def theta(self) -> np.ndarray:
        """The uniform source angles used for ``boundary`` and diagnostics."""
        return self._theta.copy()

    def __repr__(self) -> str:
        return (
            f"RiemannMap(n={self.diagnostics.n_boundary}, "
            f"F(0)={self.diagnostics.f_at_zero:.6g}, "
            f"source_rotation={self.diagnostics.source_rotation:.6g})"
        )


def riemann_map(bound, *, n: int = 1024, diagnostics: bool = True) -> RiemannMap:
    """Convenience wrapper for :meth:`RiemannMap.from_boundary`."""
    return RiemannMap.from_boundary(bound, n=n, diagnostics=diagnostics)
