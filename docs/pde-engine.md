# PDE route: boundary selection + interior flat-metric trace

This note collects the implementation details of the PDE route. The public API is
small; the diagnostics and the domain restriction are the parts worth reading
before using it as a replacement for the harmonic-measure engine.

## Boundary correspondence selection (`hbs.boundary_selection`)

`solve_sliding_boundary(curve, derivative, ...)` chooses a parameterization `p` of
a positively oriented Jordan curve `γ` so that `γ ∘ p` has no positive Fourier
modes above the first and its first mode is real positive **relative to the input
curve's own first Fourier coefficient**. This is an implicit
exterior-conformality condition; it takes no exterior density and no prescribed
boundary correspondence `β`.

The solver is a damped Gauss–Newton and is **local**. It checks sampled
monotonicity and a denser Fourier residual, not a continuous certificate. By
default it continues in Fourier order (`8 → 16 → 32 → …`) until the dense residual
is below `target_residual` or `max_modes` is reached. Supplying `initial` disables
automatic continuation and gives the caller full warm-start control.

Inspect `diagnostics["accepted"]`, `diagnostics["dense_forbidden_residual"]`, and
`diagnostics["dense_min_parameter_speed"]`; `accepted` alone is not a proof. The
module only selects the correspondence — it does not solve HBS, extend
harmonically, or reconstruct `μ`.

```python
import numpy as np
from hbs.boundary_selection import solve_sliding_boundary

curve = lambda t: 1.2 * np.exp(1j * t) + 0.1 * np.exp(-2j * t)
derivative = lambda t: 1.2j * np.exp(1j * t) - 0.2j * np.exp(-2j * t)
theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
selection = solve_sliding_boundary(curve, derivative)
boundary = curve(selection.parameter(theta))
print(selection.diagnostics["accepted"])
```

The public exports are `BoundarySelection` and `solve_sliding_boundary`.
`SlidingBoundaryProblem` and `parameter_basis` are internal helpers
(`_SlidingBoundaryProblem`, `_parameter_basis`) and are not part of the public API.

## Engine (`method="pde"`)

`method="pde"` combines the exterior one-sided Fourier selector with the interior
flat-metric PDE

    Δu = 0 in Ω,    κ + ∂ₙu = eᵘ on ∂Ω,

then applies the paper's M1 centering `Σ Φ₁⁻¹(z_k)=0` on the full collocation
grid. It does not use the harmonic-measure density.

After M1 alignment it matches the harmonic-measure engine up to a source-circle
rotation. On the development shapes `bent_lobe` / `asym_dent3` / `ellipse`:
welding rms `8e-4°` / `0.35°` / `8e-6°`, μ rms `0.0009` / `0.018` / `8e-7`.
Rotation, translation, scaling and start-point rolls are invariant.

The solver is local. Strong non-convex boundaries (for example
`cos t + 0.3 cos 2t + i(sin t + 0.2 sin 3t)`) can leave the interior Robin residual
around `0.13`; the engine then raises `RuntimeError` explicitly instead of silently
degrading. The low-level entry is `hbs.pde.pde_welding`; its supported default is
`n=128`.

### Selector continuation and warm start

The PDE route does not need the standalone selector's `2e-7` residual. It uses
`min_modes=64`, `max_modes=192`, `target_residual=3e-4`, and only allows an earlier
stop when the dense residual is already below `2e-8`. The engine gate is `3e-4`.
This keeps the low-order stages for simple shapes, advances `asym_dent3` to 128
instead of 192, and advances `three_lobe` to 64 instead of 128; measured
welding/μ changes are at most about 2% on those shapes, and zero on the others.

For `n >= 256`, `pde_welding` first solves `n=96`, periodically interpolates `u`
onto the fine grid, and uses it as the initial guess. On the development shapes
the interior iterations at `n=512` fall from 7 to 3 (analytic/ellipse/peanut/
bent) and from 11 to 4 (`asym_dent3`), with the same residual scale.

### Performance note

The public HBS grid defaults are `density=0.01` and `circle_point_num=500`; the PDE
trace `n` is independent. On six smooth shapes at
`density=0.01`, `circle_point_num=500`, forward median times were approximately
`harmonic 138 ms`, `zipper 180 ms`, `pde n=96 136 ms`, and `pde n=512 225 ms`
(single-threaded, three repetitions) before the selector/warm-start changes.
After them, `pde_welding` on the hard shape `asym_dent3` dropped from about
`286 ms` to `104 ms` at `n=96` and from `379 ms` to `164 ms` at `n=512`; simple
shapes are within timing noise because the shared `build_hbs` stage dominates the
full HBS call.

A separate `n` scan against the harmonic-measure reference found: `n=64` fails the
interior acceptance gate on `asym_dent3`; `n=80` gives asymmetric μ rms `2.1e-3`;
`n=96` gives `1.8e-3`; `n=128` gives `1.2e-3` and is the first accuracy plateau
(`n=192` and `n=512` are the same). The total `pde_welding` cost of `n=128` is only
about 3–5 ms above `n=96` on the simple shapes and is unchanged on `asym_dent3`,
because the selector dominates there. `n=128` is the only supported default;
raising `n` beyond 128 does not improve reconstruction fidelity.

The shared FEM path now also caches the `disk`/`fem` objects and the `splu(A)`
factorization in-process. `build_hbs` drops from about 134–155 ms per call to
about 68 ms after the first factorization; the full `get_hbs(method="pde")`
call is 2.2–4.1× faster than the pre-cache baseline on the six-shape set. First
call still pays one factorization (about 69 ms). The in-process caches are
bounded (LRU: 8 meshes, 8 FEM operators, 2 factorizations) so repeated custom
`DiskMesh` objects do not grow memory without limit; `hbs.cache.clear_memory_cache()`
releases them without touching the on-disk mesh/FEM cache.

A same-benchmark comparison with accuracy, speed, feasibility and invariance is
kept in `HBS-research/runs/三引擎同台对比_2026-11-04/报告.md`; the optimization
baseline and before/after tables are in
`HBS-research/runs/PDE提速_2026-11-04/报告.md`.