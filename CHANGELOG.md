# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/).

**Release status.** Published releases are listed on
[PyPI](https://pypi.org/project/hbs/#history); the `[Unreleased]` section and any
version newer than PyPI's latest describe work that is on `main` only.

## [Unreleased]

### Added — branch-free source-circle gauge `normalize="c0"` (now the default)

- `normalize="c0"` selects the source-circle rotation by the weight-1 functional
  `C₀ = M₁·conj(M₃) = (∫zμ dA)·conj(∫μ dA)`. Weight 1 means `e^{-iT}C₀ = |C₀|` has a
  **single** solution, so `c0` has no 180° branch decision at all — unlike `m34`,
  whose `T mod π` needs the `arg I₂` tie-break.
- It uses the two discretely cleanest moments (`∫μ dA` sees only the m=0 error
  component, `∫zμ dA` the m=−1 one), so its equivariance residual is
  1.3e-4–8.6e-4° on the eight real shapes — the best of the
  `C_k = M_{k+1}·conj(M_k)` family.
- Failure is self-announcing: `C₀` vanishes exactly where the phase is not
  identifiable, and `GaugeChoice.confidence_c0 = |C₀|/(S₁S₃)` collapses there.
  Below `I2_CONFIDENCE` the choice reports `kind="none"`, `phase=None`.
- Structurally, **every** weight-1 gauge must return undetermined on rotationally
  symmetric shapes (`n ≥ 2`: `F = e^{-2πi/n}F ⇒ F = 0`). `c0` therefore reports
  `none` on centrally symmetric inputs where `m34` can still give `T mod π`.
- `GaugeChoice` gained trailing defaulted fields `c0` and `confidence_c0`;
  `as_dict()` includes `"c0"` and `"confidence_c0"`.
- **The default is now `normalize="default"`, which is `c0`.** The old automatic
  `"auto"` strategy (I₂ when confident, M3+M4 otherwise) is retired: `"auto"` is no
  longer a valid value. `"i2"` and `"m34"` remain selectable and produce
  **bit-for-bit** the same output as before; `c0`/`default` use a separate summation
  path (`_c0_moments`), pinned by
  `tests/test_gauges.py::test_existing_kinds_are_bitwise_unchanged`.
- `AUTO_I2_CONFIDENCE` is kept only for import compatibility; no `normalize` value
  uses it any more.

### Changed — documentation and packaging

- `README.md` is rewritten around current behaviour: PyPI project page / release
  history / files links and version badges up top, a quickstart, the two knobs
  (`method`, `normalize`) and the opt-in `validate_reconstruction` as tables, the
  public API with signatures, the 60 796 / 62 512 face-count convention, a
  "facts that bite" section (the `1/rx²` scale law, seam rotation is not a gauge,
  start-point invariance), and a measured block for `img/example.png` on 2.3.2.
- The README's old claim that I₂ is "~1500× more sensitive to mesh discretization
  than M₃" is replaced by the number 2.3.0 actually measured: the two gauges
  converge at O(h²) with a constant **48–56×** error ratio, so refinement does not
  close the gap.
- `example.ipynb` is rebuilt and re-executed end to end with a real kernel on
  2.3.2 (15 cells, outputs embedded): boundary → forward solve with
  engine/rung/gauge diagnostics → welding, `|μ|` and extension plots →
  reconstruction scored by `fidelity_sim` → `compare_engines` cross-check → the four
  `normalize` settings side by side plus `validate_reconstruction=True`.
- `CLAUDE.md` is no longer tracked: development notes (module map, numerical
  pitfalls, workflow) stay local and out of the public repository. It was never in
  the wheel, and the published 2.0.2 sdist never contained it either; `git log`
  still holds the earlier copies.
- Section order is fixed: 2.3.1 sat **below** 2.3.0, which broke the
  newest-first rule and made the 2.3.0 → 2.3.1 compare link read backwards.

### Notes — what is actually on PyPI

Checked against `https://pypi.org/pypi/hbs/json` on 2026-10-29: the published series is
`… 1.1.0, 2.0.0, 2.0.1, 2.0.2, 2.3.1, 2.3.2` — **2.1.0, 2.2.0 and 2.3.0 were cut in git
but never uploaded**; their behaviour changes reached PyPI users inside 2.3.1. So
`pip install hbs` gives 2.3.2, and the reconstruction guard is installable rather than
`main`-only.

## [2.3.2] — 2026-10-28

### Added — optional reconstruction guard with a one-step mesh fallback

- `get_hbs(..., validate_reconstruction=True)` / `get_hbs_result(...)`: solve, then
  check that `reconstruct_from_hbs` yields a real curve; if it degenerated, redo the
  solve on `circle_point_num // 2` (floor 64) and check again; if both fail, raise
  `RuntimeError` instead of returning an unvalidated result. `HBSResult.guard_rung`
  reports the fallback (`"fallback:<circle>"`).
- Why it exists: near-critical inputs (`|μ| → 1` on many faces) make
  `geodesic_welding` collapse every exterior point onto `|z| = 1` — output is
  `circle_num` copies of `1.0`: no exception, finite, and **not a curve**. Frozen
  `shape100-v1` data: the fallback fires on 4/100 and rescues 1
  (`real_chopper-3`-shaped); `concave_quad`, `real_Bone-3`, `real_hammer-3` are
  sentinels on both meshes.
- Ported from the research repo's 1.2.0 vendored angular path
  (`src/hbs/angular.py::_reconstruct_is_valid`, fixed 2026-10-24 so the fallback
  result is re-checked); the research repo now imports `hbs` through a symlink, so
  the logic has one home.
- **Off by default** and skipped when an explicit `disk` is passed: it costs one
  extra full reconstruction (LSQC solve) and the default behaviour stays
  "always returns the mesh you asked for".

### Removed — dead vendored code

- `hbs/_angular/joint.py` (475 lines) is gone. No library path imported it;
  `import hbs` never loaded it (`hbs._angular.direct` is the engine kernel).
  Its FEM operator was a duplicate of `hbs.cache._fem_operator` — the two
  matrices were checked elementwise equal (`max|ΔA| = max|ΔB| = 0.0`) and the
  solves matched bit for bit — so it was neither a second implementation nor a
  cross-check. It remains byte-identical upstream at
  `HBS-research/src/joint_hbs_optimizer.py`, and git history keeps the copy,
  so the paper correspondence is preserved outside the distributed wheel.
  `tests/test_packaging_selfcontained.py` now only pins `direct`;
  `experiments/dense_grid_compare.py` imports the upstream module.

### Notes — `AUTO_I2_CONFIDENCE` is unchanged

- `1e-2` is **pre-set, not measured-optimal**; the comment pointing at
  `runs/规范切换_2026-10-28/` was stale (no such directory). The 2026-10-28
  equivariance scan (`experiments/auto_gate_scan.py`) is the first real data:
  on the 8 real shapes every `conf_i2` is 2.8–13.6%, so `1e-2` never fires
  there. Raising it to `0.1` would switch 5 of the 8 (those at 2.8–9.1%) to
  M3+M4; the 3 at 11–13.6% keep I₂ and its 0.4–0.5% error. It lowers the
  aggregate mean (5.61% → 2.56%) but leaves the 64.8% worst case untouched.
- `conf_i2` does not predict which gauge is more consistent: on real shapes
  I₂'s source-rotation equivariance error is **0.36–2.74% of `peak|μ|`**
  (mean 0.95%), M3+M4 is **0.0002–0.001%**. After reconstruction and
  similarity alignment the *shape* difference is 0.1–1.3% of `R_eff`
  (measured on `bent_oval_035` and `asymmetric_oval`), so the 0.5–2% is
  mainly a signature-reproducibility nuisance, not a shape-fidelity loss.
- The real risk is the failure mode, not the magnitude: any hard gate jumps
  across its threshold (`interp_w0.65` 58.7%), and M3+M4's branch flip is
  catastrophic at low `branch_margin` (64.8% / 180°). A margin-aware selector
  (mean 1.90% vs current 5.61%) is a design change for a separate evaluation —
  it rotates every real shape's code by 10.8–158.6° and would break the
  `hbs_generation` code distribution.

## [2.3.1] — 2026-10-28

### Added — forward-invariance contract tests

`tests/test_forward_invariance.py` pins what the two engines actually guarantee:

- **Rigid transforms.** Rotation, translation, scaling and a composite leave both
  the welding and `μ` alone: welding rms `6.9e-13–1.5e-12°` (angular) and
  `3.0e-11–5.7e-11°` (zipper); `μ` rms `1.3e-11–3.0e-11%` and
  `8.4e-10–1.2e-9%` of peak. Measured on four asymmetric shapes with six
  transforms. The two engines differ only by floating-point path: angular works
  in the angle domain and sits at `eps`; zipper runs the Möbius/square-root
  chain and amplifies to `~1e-9%`.
- **Start point.** Cyclic shifts by 1 / 37 / 211 points give **bitwise
  identical** `μ` on both engines, because `prepare_boundary`'s geometric anchor
  depends only on the point set.
- **Sampling density is a convergence quantity, not an invariant.** Refining the
  same analytic curve from 200 to 3000 points drives the difference to the
  finest tier down monotonically.

**Deliberately not in the suite**: per-shape convergence *rates*. Those depend on
`circle_point_num` / `density`, so pinning them in library tests turns a config
change into a false positive. One such observation — zipper on a pear-shaped
boundary stalling at 0.24% between n=500 and n=1200 at `CIRCLE=500` — is recorded
in `HBS-research/runs/前向不变性校验_2026-10-28/报告.md` §3 instead. It was
verified to be a boundary-layer difference in the welding rather than a gauge
artifact: a pure phase difference would force `|Δμ|/|μ| ≡ 2|sin δ| = 0.0428`,
and the measured median is `4.7e-4`.

## [2.3.0] — 2026-10-28

`normalize` now offers two source-circle rotation gauges, not one. The default
picks between them.

### Added — the paper's M3+M4 gauge

- `hbs.gauges.select_gauge(mu, disk, kind)` implements the paper's (5.8)+(5.9)
  normalization as a **closed form**. With `τ₀ = arg ∫μ dA` and
  `τ₁ = arg ∫μ/z dA`, let `s = (τ₁ − τ₀/2) mod 2π`; then `T = τ₀/2` when
  `s ∈ [0, π)` and `T = τ₀/2 + π` otherwise. One evaluation, no iteration.
- `M₃(Φ_Tμ) = e^{-2iT}M₃(μ)` and `I₂(Φ_Tμ) = e^{-iT}I₂(μ)` (measured ratios
  −2.000 / −1.012). That weight difference is *why* M3+M4 needs both functional
  and I₂ alone does not: weight 2 fixes `T mod π`, and I₂ picks the branch.
- `GaugeChoice` reports `kind`, `phase`, `confidence_i2`, `confidence_m3`,
  `branch`, and `branch_margin` (distance from `s` to the flip points `{0, π}`
  in radians). `HBSResult.gauge` carries it through `get_hbs_result`.

### Changed

- `normalize` accepts `"auto"` (default), `"i2"`, `"m34"`, `"none"`.
  `"auto"` uses I₂ when `|I₂|/Σ|term| ≥ AUTO_I2_CONFIDENCE = 1e-2`, otherwise
  falls back to M3+M4. The switch is deliberately **discontinuous**: across the
  gate, neighbouring shapes take different gauges and their `μ` differ by a
  rotation. `gauge.kind` reports it rather than pretending continuity.
- `build_hbs` returns `(mu, field, welding, disk, choice)`; the choice is what
  the caller needs to know whether the phase is meaningful.
- `get_conformal_welding`'s `normalize` default follows `get_hbs` (`"auto"`).

### Measured — rotation equivariance

Injecting a source-circle rotation `T` on a fixed welding (16 samples over the
full circle, six clearly rotation-asymmetric shapes, angular engine), the total
rotation `T+θ(T)` should stay constant if the gauge is equivariant:

| gauge | total-rotation spread | μ rms on the same mesh |
|---|---|---|
| I₂ | 0.895–2.196° | 1.021–2.598% |
| M3+M4 | **0.0001–0.0006°** | **0.0001–0.0013%** |

Both engines (`T* = 232.9–233.2°`) land on the same `μ` after the gauge:
M3+M4 0.109–0.539% rms, I₂ 0.378–3.859%, versus 41.8–86.8% ungauged.

Where the 50× gap comes from — measured, not assumed. Decomposing the
equivariance defect `e = μ_T − e^{−2iT}μ₀` into angular modes under area weight,
`m = 0` (what `M₃` sees) is `1.1e-6` while `m = 1` (what `I₂` sees) is `1.2e-2`:
the FEM weak solve drives the zeroth moment to roundoff, so `∫μ dA` is blind to
the defect and `∫μ/z dA` takes it whole. It is *not* "`1/z` amplifies the error
near the origin" — the `m = 1` mass at `r < 0.5` is 8× smaller than at
`r ∈ [0.5, 0.9)`. Both gauges converge at O(h²) under mesh refinement
(`density` 0.02 → 0.005) with the 48–56× ratio constant, so refinement does not
close the gap either. Nor is it anchor weakness: the spread scales as
`conf^−0.23`, not `conf^−1`. Probes live in `runs/焊接规范不变性_2026-10-28/`.

### Fixed — the old M3+M4 implementation

`src/hbs/hbs.py` in the 1.2.0 vendored tree applied the paper's normalization as
`rotate_x(r / 2)` inside a 30-iteration loop, with oscillation detection and a
`raise RuntimeError` on non-convergence. Two defects:

1. `r` is recomputed from the **current** residual each round, so rotating by
   `r/2` shrinks it by half per iteration instead of eliminating it — the loop
   only converges because `30` rounds of halving is enough. The paper's (5.10)
   is a closed form.
2. The branch test used `arg Σμ·z` (first moment `M₁`), which does not appear in
   the paper. (5.9) branches on `arg ∫B/z dA`, i.e. `I₂`.

### Fixed — compatibility wrapper

- `hbs.angular.angular_mu_from_boundary` was still unpacking `build_hbs`'s
  four-tuple return. Its `normalize` default is now `"auto"` and its `stats`
  dict carries the `GaugeChoice` as `"gauge"`. No test covered this path, which
  is why the tuple change did not surface; `tests/test_gauges.py` now does.

### Notes

- `AUTO_I2_CONFIDENCE = 1e-2` is a first cut, not a scanned optimum. All six
  shapes measured sit at 2.7–5.7%, so `"auto"` ≡ `"i2"` on them; the fallback
  only engages on near-symmetric shapes.
- `i2_phase`, `apply_i2`, and `_phase_from_arrays` keep their 2.2.0 signatures.

## [2.2.0] — 2026-10-28

`method` now really switches engines. Before this release both values ran the
same welding code, so `method="zipper"` was a fallback switch, not the zipper
algorithm — the two engines could not cross-check each other.

### Changed — architecture

- **Two independent engines, no mutual fallback.** New `hbs/_engines.py` holds
  `angular` (boundary-layer logarithmic single-layer potentials) and `zipper`
  (Bishop/Marshall zipper) behind one registry. Each runs only its own input
  ladder; if an engine fails on every rung it raises, it never switches to the
  other. Measured agreement: the two weldings differ only by a source-circle
  rotation, `rms(δ−S) = 0.08°` at N=500, shrinking like O(1/N).
- **One shared backend.** New `hbs/_pipeline.py` owns everything after the
  welding: prepare (orient → geometric anchor → arclength resample), FEM
  extension, `μ = H_z̄/H_z`, and the I₂ gauge. Previously `angular` had its own
  `_mu_on_mesh` and `zipper` had a separate `_finalize`, so the same
  normalization existed twice and inevitably drifted.
- **New `get_hbs_result`** returns an `HBSResult` with `engine` and `rung`, so
  callers can tell which engine ran and whether a degraded resample was used.
  `get_hbs` keeps its four-tuple contract.
- **New `compare_engines`** compares two engine outputs *after* removing the
  source-circle rotation. Doing this by hand is a trap: feeding each engine's
  own `y` into the FEM reports `μ` relative difference 0.80, while removing the
  rotation first gives 0.006.

### Removed

- `hbs.angular._mu_on_mesh`, `_solve_with_mesh_fallback`, `_reconstruct_is_valid`,
  `_mesh_for`, `_fem_for`, `_default_mesh`, `_FALLBACK_CIRCLE`: the 512→256 mesh
  fallback is gone with them. The pipeline does not retry on a coarser mesh;
  near-critical shapes raise from the engine.
- `normalize="first-moment"`: it was the angular solver's intrinsic phase and
  had no counterpart on the zipper path. Only `"i2"` and `"none"` remain.
  Passing `"first-moment"` raises `ValueError`.
- `hbs/qc/bc.py` and `hbs/utils/tool_functions.py`: pure compatibility shims
  predating 2.0.
- `get_conformal_welding` no longer has a private zipper branch; it uses the
  same engine registry as `get_hbs` and therefore returns bitwise-identical
  `(x, y)` for the same `method`.

### Fixed

- `reconstruct_from_hbs` sliced the interior points as
  `mapping[circle_num : circle_num+in_vert_num]` and passed that to
  `geodesic_welding`, which expects `[boundary; inner]`. Boundary and interior
  points were misaligned, so the welded boundary replaced the LSQC boundary
  only by coincidence. Now the boundary prefix is included, matching what the
  function has always documented.
- `hbs/utils/poisson.py` is annotated as experiment-only. Library code uses
  `hbs.extension.fem_extend`; the dense Poisson kernel remains only so the
  historical comparison scripts in `experiments/` stay reproducible.

### Notes

- `hbs/_angular/joint.py` (475 lines) is no longer used by any library code
  path; only `experiments/dense_grid_compare.py` and the packaging test refer
  to it. It is vendored upstream code, so removing it would break the paper
  correspondence — left in place, flagged here.

## [2.1.0] — 2026-10-28

`method="zipper"` becomes **exactly invariant** under the shape symmetries that
matter for a signature. μ fields are now bitwise identical under start-point
rolls and orientation reversal, and agree to ~2e-10 under rotation, scale and
translation. Reconstruction inherits the same invariance.

### Changed — numerical

- **The 8-round λ loop is gone.** `get_hbs(method="zipper")` used to iterate
  `cw.rotate_x(r)` → `linear_interp` → recompute the field, stopping at
  `|r| ≤ 5e-3`. `rotate_x` rotates the source-circle parametrization, and the
  `linear_interp` that followed always resampled onto a **fixed** x grid
  starting at angle 0 — so unless `r` was an exact multiple of the grid step
  this was a resampling, not a rigid rotation. The stop-gate residual therefore
  depended on the input start point, and the seam amplifies `dμ/dr` by ~125×
  (measured), turning a 5e-3 residual into ~0.6 of |μ|. Measured roll residual
  on the old path: 0.46–0.87.
  Replacement: a single conjugation `μ ← e^{-2iθ}μ` with mesh vertices
  multiplied by `e^{-iθ}` (θ = arg I₂), via `hbs.gauges.apply_i2`. No re-solve,
  no re-interpolation. `_LAMBDA_MAX_ITERATIONS` and `_lambda_phase` no longer
  exist.
- **`normalize` now means the same thing on both paths.** `"first-moment"`
  (default) degenerates to the I₂ gauge on the zipper path; `"i2"` is the
  identical one-shot operation; `"none"` returns the raw engine output. An
  invalid value now raises `ValueError` instead of being silently ignored.

### Fixed — zipper start-point invariance

- `_resample_boundary` anchors the arclength grid at `bound[0]`, so rolling the
  input changed the **resampled point set**, not just its cyclic labelling.
  Reordering resample/reversal cannot fix that (measured: 3e-3–7e-2 pointwise
  difference either way). `_prepare_boundary` now does orientation → geometric
  anchor → resample, where the anchor is the vertex farthest from the centroid,
  with a turning-angle tie-break (also rotation-invariant; `argmax Re` is not).
  Roll residual afterwards: **bitwise zero**.
- The reversal (`_signed_area > 0 → bound[::-1]`) that conditioning requires is
  applied *before* the anchor, so both winding orders of the same shape land on
  the same anchor.
- `get_conformal_welding(method="zipper")` shares the same `_prepare_boundary`
  and the same kernel (`zipper_welding_prepared`) as `get_hbs`. Before this, the
  two public entry points disagreed: sorted `x` after a roll differed by 5.56.

### Added

- `tests/test_zipper_invariance.py` — 15 contract tests: bitwise roll and
  reversal invariance, rotation/scale/translation to fp precision, I₂
  magnitude preservation, agreement between the two public entry points, and
  guards that the iteration constant stays deleted.
- `hbs.hbs._prepare_boundary`, `_canonical_start_index`, `_turning_angles`;
  `hbs.conformal_welding.zipper_welding_prepared`.

### Scope

The invariance contract is stated for **asymmetric, well-conditioned shapes**
(no rotational symmetry, `max|μ| < 1`). Shapes with exact rotational symmetry
have multiple equidistant anchor candidates; the anchor is still deterministic
but is only a discrete choice among equivalent tips.

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

[2.3.2]: https://github.com/ChanceAroundYou/hbs_python/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/ChanceAroundYou/hbs_python/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/ChanceAroundYou/hbs_python/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/ChanceAroundYou/hbs_python/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/ChanceAroundYou/hbs_python/compare/v2.0.2...v2.1.0
[2.0.2]: https://github.com/ChanceAroundYou/hbs_python/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/ChanceAroundYou/hbs_python/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/ChanceAroundYou/hbs_python/compare/v1.1.0...v2.0.0
