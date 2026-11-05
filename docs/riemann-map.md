# Riemann maps from harmonic measure

`RiemannMap` computes a numerical Riemann map

\[
F:\mathbb D\to\Omega
\]

for a smooth simply connected Jordan domain \(\Omega\).  It uses the **interior
harmonic-measure density** (the normal derivative of the Green function) to
find the conformal boundary correspondence, then evaluates the map by the
Cauchy integral.  It does **not** use harmonic extension and it does **not**
treat the HBS parent map \(H:\mathbb D\to\mathbb D\) as a conformal map.

## Construction

Let \(G(z)=G_\Omega(z,z_0)\) be the Green function and \(H\) its harmonic
conjugate.  Then

\[
\Phi=G+iH,\qquad \Psi=e^{-\Phi}
\]

is holomorphic, maps \(\Omega\) onto \(\mathbb D\), and its inverse is the
Riemann map.  On the boundary \(G=0\), and

\[
q(t)=\partial_n G(\gamma(t))\,|\gamma'(t)|
=\frac{d}{dt}H(\gamma(t)),
\]

where \(q\) is the interior harmonic-measure density.  Integrating \(q\) gives
the circle phase \(\alpha(t)\), hence

\[
F(e^{i\alpha(t)})=\gamma(t).
\]

The boundary data are evaluated by

\[
F(w)=\frac{1}{2\pi i}\int_{|\zeta|=1}
\frac{f(\zeta)}{\zeta-w}\,d\zeta,
\qquad
F'(w)=\frac{1}{2\pi i}\int_{|\zeta|=1}
\frac{f(\zeta)}{(\zeta-w)^2}\,d\zeta .
\]

This is a Cauchy integral of holomorphic boundary data, not a Poisson
(harmonic) extension.

## API

From boundary samples:

```python
import hbs

rm = hbs.RiemannMap.from_boundary(bound, n=1024)
z = rm(0.4 + 0.2j)
dz = rm.derivative(0.4 + 0.2j)
print(rm.diagnostics.f_at_zero, rm.diagnostics.f_prime_at_zero)
```

From an analytic parameterization:

```python
rm = hbs.RiemannMap.from_curve(curve, derivative, n=1024)
```

`curve(t)` and `derivative(t)` must be vectorized over a 1-D real array and may
use either orientation; clockwise input is reversed internally.  Both
constructors return the same `RiemannMap` object.

Convenience:

```python
rm = hbs.riemann_map(bound, n=1024)
```

## Normalization

The harmonic-measure solve fixes the M1 normalization.  The target point
`F(0)` is reported in `rm.diagnostics.f_at_zero`; it is **not assumed to be
zero**.  A source-circle rotation

\[
F_{\rm normalized}(w)=F(e^{i\phi}w),
\qquad
\phi=-\arg F'(0)
\]

is applied so that `F'(0)` is positive real.  This changes only the source
parameterization, not the target domain.

## Diagnostics

`RiemannMap.cauchy_riemann_residual(grid=81, radius=0.6)` evaluates a
five-point finite-difference check on a square grid in the source disk and
returns:

```python
{
    "grid": 81,
    "radius": 0.6,
    "h": 0.015,
    "max_abs_Fzbar": ...,
    "max_abs_mu": ...,
    "min_abs_Fz": ...,
}
```

`max_abs_Fzbar` and `max_abs_mu` are numerical finite-difference residuals,
not a continuous proof.  The constructor also reports the minimum phase step,
the minimum raw interior density, `F(0)`, `F'(0)`, and the source rotation.

## Scope and limits

- Requires a smooth positively oriented Jordan boundary with strictly positive
  interior harmonic-measure density.  Corners, cusps, pixel staircases and
  rough contours can fail explicitly.
- Long thin domains suffer harmonic-measure crowding; use adaptive boundary
  sampling or a different numerical conformal-mapping method.
- The boundary correspondence is sampled on `n` uniform source angles.  A
  cusp can be under-resolved even when the interior mesh is fine.
- Multiply connected domains are not covered: there is no single Riemann map
  from the disk.
- `F` is the Riemann map.  The HBS map `H` and its Beltrami coefficient
  \(\mu_H\) are different objects.

## Validation

The public prototype is checked against the known univalent map
\(f_0(z)=z+0.15z^3\) with a nonlinear boundary reparameterization.  At
\(n=1024\), the research probe reports

```text
interior max error vs f0        ~ 1.2e-10
max |F_zbar| (five-point)       ~ 5e-14
max |F_z - f0'|                 ~ 1e-9
min |F_z|                       ~ 0.79
```

These are finite-mesh readings for a smooth synthetic domain, not a general
continuity or convergence theorem.
