# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/).

## [2.0.2] — 2026-10-27

Documentation links only. **No change to any numerical result.**

### Fixed

- `README.md` now links `CHANGELOG.md` and `LICENSE` with absolute GitHub
  URLs. Relative links render fine on GitHub but are dead on PyPI, whose
  project page is a single-page app that answers every path with 200 — so the
  sole entry point to the changelog silently went nowhere.

## [2.0.1] — 2026-10-27

Documentation and metadata only. **No change to any numerical result.**

### Fixed

- Corrected a claim made in 2.0.0's README, `get_hbs` docstring and
  `hbs/angular.py` module docstring: `method="zipper"` reproduces 1.1.0's
  *welding solve path and return contract*, **not its numbers**. 2.0.0 replaced
  the λ-normalization with the area-weighted `I₂` phase, which lands on a
  different seam rotation; reconstruction fidelity on ellipse-like and
  lobed shapes differs by 10–100× from 1.1.0. To reproduce a 1.1.0 reading,
  install `hbs==1.1.0`.
- Moved the 2.0.0 solver comparison, benchmarks and migration notes out of
  `README.md` into this file. The README now documents only current behaviour.

## [2.0.0] — 2026-10-27

### Changed — BREAKING

- `get_hbs` gains `method`, defaulting to `"angular"` (angular-velocity
  forward solve + FEM field transfer). The previous conformal-welding path is
  still available as `method="zipper"`.
- `method="angular"` returns `cw=None` and ignores `circle_point_num`/`density`,
  using a fixed `(density=0.04, circle=512)` mesh (4280 faces). A `UserWarning`
  is emitted if a non-default value is passed without an explicit `disk`.
  Pass `disk=` to control the mesh.
- The `(hbs[face], field, cw_or_None, disk)` return contract is unchanged, and
  `reconstruct_from_hbs` was not touched — the backward path is identical.

### Added

- Failure-only input ladder for the angular solver. A raw boundary that fails
  is retried as `resample(500) → +Gaussian σ=2 → +Gaussian σ=4`; the first rung
  that passes the legality gate is returned and reported in
  `stats["boundary_rung"]` / `stats["input_smoothed"]`, with a `UserWarning`.
  If every rung fails a `RuntimeError` is raised (with the last underlying error
  chained) instead of returning a degenerate curve.
- `ConformalWeldingNumericalError`, raised when the zipper pipeline cannot
  produce a usable circle correspondence; the exception no longer leaks an
  `argmin(empty)`.
- `hbs.utils.eval.fidelity_sim` — similarity- and orientation-invariant shape
  fidelity.
- `tests/test_packaging_selfcontained.py`: 7 packaging guards.

### Changed — numerical

- λ-normalization: `angle(Σμ)` + `rotate_x(r/2)` + up to 30 iterations
  (1.1.0) became the area-weighted `I₂ = Σ μ·|A|/z` phase + `rotate_x(r)` +
  up to 8 iterations (`_LAMBDA_MAX_ITERATIONS`). `I₂` rotates as
  `e^{-iθ}`, so it is a continuous single-valued phase with no 180° flip.
  A confidence gate (`|I₂|/Σ|term| < 1e-4`) returns "untrustworthy" instead of
  reporting a noisy near-zero phase, and an oscillation-divergence early break
  handles symmetric/near-symmetric shapes — those return the current GHBS
  representative instead of raising.
- Boundary resampling to 500 points is now unconditional (1.1.0 only resampled
  boundaries shorter than 400 points).
- Fixed an operator-precedence bug in the old NaN exclusion,
  `np.isnan(hbs) + norm(face_center, axis=1) == 0`, under which NaN faces were
  never actually excluded.

### Fixed — packaging

- **Self-contained package.** The forward engine is vendored into
  `hbs/_angular/` (`direct.py`, `joint.py`, NumPy/SciPy only). The wheel no
  longer imports research-repo top-level modules, so the default path works
  from a clean install.
- **Python 3.8/3.9 annotations.** Seven modules gained
  `from __future__ import annotations`. The published 1.1.0 raised `TypeError`
  on import under 3.8 despite declaring support.
- **Warning-free import.** A module-level `warnings.warn` experimental module is
  no longer imported at package top level, so `pytest -W error` downstream
  works.
- **GPL containment.** `angle_slit_welding.py` (a verbatim port of the Marshall
  zipper program) was moved out of the package into a gitignored
  `unpublished_gpl/` directory; it is not in the wheel. A test guards this.
- **METADATA.** Added the `LICENSE` file to the wheel, `License: MIT`,
  Python 3.8–3.13 classifiers, `numpy>=1.20` / `scipy>=1.5` lower bounds.
  `requirements.txt` was reduced to the hard dependencies plus `pytest`.
- Optional extras: `hbs[plot]` (matplotlib), `hbs[boundary]` (opencv-python),
  `hbs[all]`.

### Measured — angular vs zipper

Both engines driven through the packaged entry points and scored with the same
evaluator, on the frozen `shape100-v1` set (legal = the reconstruction is a
simple closed curve):

| | `angular` (default) | `zipper` |
|---|---|---|
| legal end-to-end | **90/100** | 91/100 (5 more solve but self-intersect) |
| median wall clock | **0.67 s** | 2.24 s |
| median Chamfer, 86 common | **3.84e-05** | 4.84e-04 |
| median excess over sampling floor | **1.42×** | 25.8× |

Angular wins the Chamfer comparison on 81 of those 86 cases (p = 9.6e-19).

On real MPEG-7 `cv2.findContours` outlines (usable = similarity- and
orientation-invariant fidelity < 0.30):

| sample | raw angular | + ladder | Zipper |
|---|---|---|---|
| 95 cases, design probe | 68 | **72** | 69 |
| 55 cases, re-checked through the packaged wheel | 37 | **39** | 38 |

On cases both engines solve, angular fidelity is far lower (median 0.0019 vs
0.1371). Rescued ladder rungs land at 0.049–0.157 — usable but degraded, hence
the warning. The remaining outlines (22/95 in the probe, 16/55 in the recheck)
are rejected by both engines: negatively oriented harmonic-measure density with
no smooth reparameterization that fixes it.

### Migration from 1.x

- Calls to `get_hbs` that need the old welding path must pass
  `method="zipper"` explicitly. Its return contract is unchanged, but its
  numbers are **not** 1.1.0's — see 2.0.1 above.
- `reconstruct_from_hbs` kept its three-argument signature; the experimental
  `post_deepen` parameter was removed entirely.
- Nothing else in the public API changed.

[2.0.2]: https://github.com/ChanceAroundYou/hbs_python/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/ChanceAroundYou/hbs_python/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/ChanceAroundYou/hbs_python/compare/v1.1.0...v2.0.0
