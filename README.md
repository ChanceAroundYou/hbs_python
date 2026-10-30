# HBS — Harmonic Beltrami Signature

[![PyPI](https://img.shields.io/pypi/v/hbs.svg)](https://pypi.org/project/hbs/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hbs.svg)](https://pypi.org/project/hbs/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ChanceAroundYou/hbs_python/blob/main/LICENSE)

**PyPI project page:** <https://pypi.org/project/hbs/> ·
**release history:** <https://pypi.org/project/hbs/#history> ·
**files:** <https://pypi.org/project/hbs/#files> ·
**source:** <https://github.com/ChanceAroundYou/hbs_python>

A Python library for the **Harmonic Beltrami Signature**: a 2-D shape is encoded by a
single complex vector — the Beltrami coefficient `μ = H_z̄/H_z` of the harmonic map
that carries the unit disk onto the shape — sampled at the centres of a triangulated
disk. `μ` is invariant under rotation, translation, uniform scaling and the starting
point of the boundary sampling, and `reconstruct_from_hbs` recovers the shape from it.

The package covers the whole loop — image → boundary → conformal welding → harmonic
extension → `μ` → LSQC reconstruction — plus the meshes, on-disk caches, source-circle
rotation gauges and comparison metrics around it.

## Reference

**Harmonic Beltrami Signature: A Novel 2D Shape Representation for Object
Classification** — Chenran Lin, Lok Ming Lui.
DOI: [10.1137/22M1470852](https://doi.org/10.1137/22M1470852)

## Installation

```bash
pip install hbs              # core: NumPy + SciPy only
pip install "hbs[plot]"      # + matplotlib     (hbs.plotting)
pip install "hbs[boundary]"  # + opencv-python  (hbs.utils.boundary)
pip install "hbs[all]"       # both
```

Python 3.8 – 3.13. NumPy ≥ 1.20 and SciPy ≥ 1.5 are the only hard dependencies;
matplotlib is imported lazily inside `hbs.plotting`, so `import hbs` never pulls it in
and `pytest -W error` downstream stays clean.

From source:

```bash
git clone https://github.com/ChanceAroundYou/hbs_python.git
cd hbs_python
pip install -r requirements.txt   # hard deps + pytest; extras listed commented out
pip install -e .
```

## Quickstart

```python
from hbs import get_hbs, reconstruct_from_hbs
from hbs.utils.boundary import get_boundary   # needs hbs[boundary]

bound = get_boundary("img/example.png", 250)          # n x 2, either winding order
mu, field, welding, disk = get_hbs(bound)             # method="angular", normalize="default"

rec_bound, in_points, out_points, mapping = reconstruct_from_hbs(mu, disk)
```

`disk` **must** be the object `get_hbs` returned: the gauge rotates the mesh vertices,
so it hands back a *new* `DiskMesh`, and a face-count mismatch makes LSQC assert.

## The two knobs

`get_hbs(bound, circle_point_num=500, density=0.01, disk=None, method="angular",
normalize="default", validate_reconstruction=False)`

### `method` — which forward engine solves the welding

| `method` | algorithm | character |
|---|---|---|
| `"angular"` (default) | boundary-layer logarithmic single-layer potentials | most accurate on smooth curves; needs a strictly positive interior/exterior harmonic-measure density |
| `"zipper"` | Bishop/Marshall zipper | works from the point sequence alone; more tolerant of irregular sampling; structurally degraded on highly symmetric shapes |

The two are **independent implementations and never fall back to each other**. An
engine that fails retries only inside its own input ladder

```
raw → resample(500) → resample(500) + Gaussian σ=2 → resample(500) + Gaussian σ=4
```

and raises `RuntimeError` (with `__cause__` chained) if every rung fails — it will not
return a degenerate curve, and will not switch engines. The rung that succeeded is
reported in `HBSResult.rung`: anything other than `"raw"` is a **degraded** result
(usable, but with fidelity an order of magnitude worse), so surface it to your users.

Both engines share one backend: orientation, geometric anchor and arc-length
resampling (`_pipeline.prepare_boundary`, **before** the welding), then FEM harmonic
extension, `μ` and the gauge (`_pipeline.build_hbs`, **after** it). Each engine hands
back its boundary values on the uniform source circle, so the backend never
re-samples them — and there is no second copy of that logic.

### `normalize` — which source-circle rotation gauge is applied

| value | gauge | note |
|---|---|---|
| `"default"` (default) | branch-free anchor `C₀ = M₁·conj(M₃) = (∫zμ dA)·conj(∫μ dA)` | weight 1, so `T = arg C₀` has a **single** solution — no 180° branch at all. Returns `kind="none"` / `phase=None` exactly where `C₀` vanishes; every weight-1 gauge must do so on rotationally symmetric shapes |
| `"i2"` | single anchor `I₂ = ∫ μ/z dA` | continuous, no branch; the direction is untrustworthy on near-symmetric shapes and is ~50× more sensitive to mesh discretization than `M₃` |
| `"m34"` | paper (5.8)+(5.9), closed form | `M₃ = ∫ μ dA` fixes `T mod π`, `I₂` picks the branch — one evaluation, no iteration |
| `"c0"` | same branch-free anchor as `"default"` | explicit spelling of `"default"` |
| `"none"` | nothing | raw engine output |

`GaugeChoice` carries `kind`, `phase`, `i2`, `m3`, `confidence_i2`, `confidence_m3`,
`branch`, `branch_margin` (distance to the M3+M4 branch flip, in radians), `c0`,
`confidence_c0` and a human-readable `reason`. `kind == "none"` with `phase is None`
means the functional was not trustworthy and no rotation was applied — then the
orientation of `μ` has no gauge meaning and must not be compared across shapes.

### `validate_reconstruction` — opt-in guard (since 2.3.2)

`False` by default. When enabled, the result is reconstructed once and checked: a
near-critical field (`|μ| → 1` on many faces) collapses `geodesic_welding` onto
`|z| = 1`, producing `circle_num` copies of `1.0` — finite, exception-free, and **not a
curve**. On such a collapse the solve is redone once on `circle_point_num // 2`
(floor 64); if both fail it raises `RuntimeError` rather than returning an unvalidated
signature. The landing point appears in `HBSResult.guard_rung`
(`"fallback:<circle>"`). It costs one extra LSQC solve, and is skipped when you pass an
explicit `disk`.

## Public API

`import hbs` exposes:

| call | what it gives you |
|---|---|
| `get_hbs(bound, circle_point_num=500, density=0.01, disk=None, method="angular", normalize="default", validate_reconstruction=False)` | `(mu, field, welding, disk)` |
| `get_hbs_result(...)` | `HBSResult(mu, field, welding, disk, engine, rung, gauge, guard_rung)` |
| `reconstruct_from_hbs(mu, disk, eps=0.0)` | `(bound_points, in_points, out_points, mapping)` |
| `get_conformal_welding(bound, method="angular", normalize="default", resample_to=500)` | `ConformalWelding` — bitwise identical to the welding `get_hbs` used for the same `method` |
| `compare_engines(first, second, disk, fem, circle_num)` | `EngineAgreement(welding_shift, welding_rms, mu_max, mu_rms)` — aligns the source-circle rotation first |
| `select_gauge(mu, disk, kind="default")` | `GaugeChoice` |
| `compare_welding` / `compare_conformal_welding` / `welding_curve` | welding agreement as a degree-1 circle homeomorphism |
| `get_beltrami_coefficient`, `lsqc_solver`, `mu_chop` | quasiconformal primitives |
| `get_unit_disk`, `get_rect`, `get_unit_disk_in_rect`, `Mesh`, `DiskMesh` | meshes |

`mu` is complex, `field`/`mapping` are real `n × 2` arrays; `hbs.utils.cast.to_real` /
`to_complex` convert between the two representations.

### Length convention

The HBS vector length is the face count of the mesh: the default
`get_unit_disk(0.01, 500)` gives **60 796** faces, `get_unit_disk(0.01, 1000)` gives
**62 512**. Both engines share this single default mesh since 2.1.0. Reconstructed
boundary resolution follows `disk.circle_num`. `hbs.cache.load_unit_disk(density,
circle_point_num)` returns `(disk, fem)` and caches both on disk.

## Measured on `img/example.png` (hbs 2.3.3, default mesh, 2026-10-30)

```
engine=angular  rung=raw  faces=60796  circle=500   solve 0.21 s warm (mesh cached)
gauge  default (c0)  phase 1.07°  conf_i2=0.0023  conf_m3=0.2019  branch_margin 90°
|mu|   max 0.99996, 220 of 60796 faces above 0.99
reconstruction  similarity-invariant fidelity 0.0332   (zipper on the same input: 0.2568)
               reconstruct_from_hbs 1.30 s
engines (same mesh, normalize="none")  shift 201.27°, welding rms 0.2868°,
                                       mu rms 0.0044, mu max 0.1118
```

`example.ipynb` reproduces the solve, gauge, `|μ|`, fidelity and engine-agreement
numbers above from executed cells. The two timings and the zipper fidelity come from
separate warmed runs (the notebook measures no time).

`conf_i2 = 0.0023` is below the `AUTO_I2_CONFIDENCE = 1e-2` threshold 2.3.2 used, so
2.3.2 fell back to M3+M4 here (phase 176.66°); 2.3.3 reports the branch-free `c0`
anchor instead (phase 1.07°). Similarity-invariant readings do not move with the
gauge — 2.3.2 scores `0.0174` for `auto`, `i2`, `m34` and `none` alike. The fidelity
difference **is** real and comes from the angular engine: 2.3.2 resampled the 250
input points to 500 before solving, 2.3.3 keeps them. Over the 100-shape frozen set
that change wins 81:9, but this shape sits at `max|μ| = 0.99996` where the
reconstruction is sensitive, and it is one of the minority. `mu_max` in the engine
comparison is dominated by the boundary layer (`r > 0.97`), where `μ = H_z̄/H_z`
amplifies the input condition number — judge engine agreement by `mu_rms`. Larger
benchmarks (100-shape frozen sets, MPEG-7 outlines, wall clocks, Chamfer distances)
are recorded in [CHANGELOG.md](https://github.com/ChanceAroundYou/hbs_python/blob/main/CHANGELOG.md).

## Facts that bite

1. **`|μ| < 1` is a hard bound.** `mu_chop(bound=0.9999, eps=1e-6)` saturates it
   smoothly and monotonically; values at or above 1 make the reconstruction ill-posed.
2. **`μ` is scale invariant, so the reconstruction's scale comes from the landmarks.**
   `reconstruct_from_hbs` pins disk vertices `(1,0)`/`(0,0)` to the target, forcing the
   recovered shape's x half-axis to `1.0`; the area ratio to the original is `1/rx²`.
   The shape itself is faithful up to that uniform factor (multiply by `rx` afterwards
   for absolute scale). This is a property of the μ representation, not a welding
   pathology — never judge usability with the raw area ratio, always with a
   similarity- and orientation-invariant fidelity (`hbs.utils.eval.fidelity_sim`).
3. **A seam rotation is not a gauge choice.** Rotating the seam changes the
   reconstruction itself (similarity-invariant residual 0.13–0.55, against a noise floor
   of ~1e-12), so where a normalization lands is a solver decision that has to be
   verified — not a free convention.
4. **Start point and winding order do not matter.** The pipeline normalizes to
   clockwise and anchors the resampling geometrically (farthest vertex from the
   centroid, turning-angle tie-break), so cyclic shifts give bitwise identical `μ`.
   Sampling *density* is a convergence quantity, not an invariant.
5. **`lsqc_solver` needs the matching mesh and landmarks.** Landmark vertex indices map
   to target positions; `eps > 0` regularizes sharp corners where `|μ| → 1`.

## Main modules

- `hbs.py`: `get_hbs` / `get_hbs_result` / `reconstruct_from_hbs`
- `_engines.py`: the two forward engines behind one registry (`angular`, `zipper`); each
  resamples its own output onto the uniform source circle, so the backend never does
- `_pipeline.py`: the single backend — prepare, FEM extension, `μ`, gauges,
  `compare_engines`; consumes boundary values, not a welding object
- `_guard.py`: optional reconstruction validity gate (2.3.2)
- `angular.py` + `_angular/direct.py`: the angular-velocity engine and its vendored kernel (NumPy/SciPy only)
- `conformal_welding.py`: `ConformalWelding` / `get_conformal_welding`
- `extension.py`: P1 finite-element harmonic extension, shared by both engines
- `cache.py`: on-disk cache for the disk mesh and its FEM operator
- `gauges.py`: I₂ anchor, the paper's M3+M4 closed form, and the branch-free `c0` anchor
- `weld_compare.py`: welding agreement metric (degree-1 circle homeomorphism)
- `mesh.py`: `Mesh` / `DiskMesh` / `get_rect` / `get_unit_disk` / `get_unit_disk_in_rect`
- `plotting.py`: optional plotting helpers (matplotlib imported lazily)
- `qc/`: `beltrami.py` (Beltrami coefficient, `mu_chop`), `lsqc.py` (least-squares quasiconformal map)
- `utils/`: `boundary.py` (OpenCV boundary extraction), `cast.py`, `eval.py`
  (`fidelity_sim`), `geodesic_welding.py`, `mobius.py`, `zipper.py`, and `poisson.py`
  (**experiment reference only** — library code uses `extension.fem_extend`)

## Worked example

[`example.ipynb`](https://github.com/ChanceAroundYou/hbs_python/blob/main/example.ipynb)
runs the full loop on `img/example.png` with executed outputs: boundary extraction,
forward solve with engine/rung/gauge diagnostics, welding and `|μ|` plots,
reconstruction scored by `fidelity_sim`, the two engines cross-checked after removing
the source-circle rotation, and the four `normalize` settings side by side.

## Development

```bash
python -m pytest tests/ -q                          # regression base + packaging guards
python -m pytest tests/test_packaging_selfcontained.py -q   # wheel self-containment guards
```

`requirements.txt` lists the hard dependencies plus `pytest`. Tests cover the forward
invariance contracts (rigid transforms, start-point rolls), gauge selection, the
reconstruction scale law `1/rx²`, engine agreement, the reconstruction guard, and the
packaging boundary (no research-repo imports, Python 3.8-safe annotations, warning-free
import, GPL code kept out of the distribution).

## Documentation

- [CHANGELOG.md](https://github.com/ChanceAroundYou/hbs_python/blob/main/CHANGELOG.md) — per-version changes, solver comparisons, measured benchmarks.
- [docs/reconstruction-notes.md](https://github.com/ChanceAroundYou/hbs_python/blob/main/docs/reconstruction-notes.md) — design notes on normalization and the known limits of concave reconstruction.
- [PyPI](https://pypi.org/project/hbs/) — published releases and files.

## License

MIT — see [`LICENSE`](https://github.com/ChanceAroundYou/hbs_python/blob/main/LICENSE).

## Contributing

Contributions are welcome via pull requests.

## Code Author

- Chenran Lin
