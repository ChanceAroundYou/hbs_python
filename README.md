# HBS - Harmonic Beltrami Signature

This is a Python library for computing Harmonic Beltrami Signature(HBS). It provides a set of tools for boundary conditions, mesh generation, conformal welding and other mathematical problems, particularly suited for numerical computations in complex analysis.

## Reference
This implementation is based on the paper:

**Harmonic Beltrami Signature: A Novel 2D Shape Representation for Object Classification**  
Chenran Lin, Lok Ming Lui
DOI: [10.1137/22M1470852](https://doi.org/10.1137/22M1470852)

## Installation

Install directly from PyPI:
```bash
pip install hbs
```

Python 3.8 or newer. NumPy and SciPy are the only hard dependencies.

Or install from source:
1. Clone this repository:
   ```bash
   git clone https://github.com/ChanceAroundYou/hbs_python.git
   ```
2. Install dependencies and the package:
   ```bash
   pip install -r requirements.txt   # numpy + scipy (+ pytest, for the tests)
   pip install -e .
   ```

`requirements.txt` lists only the hard dependencies plus `pytest`. The optional
extras are shown commented out there; install them via
`pip install "hbs[plot]"`, `hbs[boundary]` or `hbs[all]` when you need plotting
or OpenCV-based boundary extraction.

## Main Modules

- `hbs.py`: HBS algorithm for computing HBS and recontructing shape from HBS
- `angular.py`: Angular-velocity forward solver (the 2.0.0 default), plus the
  failure-only input ladder
- `_angular/`: Vendored angular-velocity and harmonic-extension engine
  (`direct.py`, `joint.py`); depends only on NumPy/SciPy
- `conformal_welding.py`: Conformal welding algorithm implementation
- `mesh.py`: Mesh generation and processing
- `plotting.py`: Optional plotting helpers (matplotlib is NOT required for compute)
- `utils/`:
  - `boundary.py`: Boundary processing tools (requires opencv-python)
  - `cast.py`: Complex ↔ real array conversion
  - `eval.py`: Similarity-invariant shape fidelity (`fidelity_sim`)
  - `geodesic_welding.py`: Geodesic welding
  - `mobius.py`: Möbius transformations
  - `poisson.py`: Poisson integral implementation
  - `zipper.py`: Zipper algorithm implementation
- `qc/`: Quasiconformal mapping algorithms
  - `beltrami.py`: Beltrami coefficient computation + mu_chop
  - `lsqc.py`: Least squares quasiconformal mapping algorithm

Top-level `import hbs` exposes the full public API:
`get_hbs`, `angular_mu_from_boundary`, `reconstruct_from_hbs`,
`get_beltrami_coefficient`, `lsqc_solver`, `mu_chop`, `get_conformal_welding`,
`get_unit_disk`, `get_rect`, `get_unit_disk_in_rect`, `Mesh`, `DiskMesh`.

Backward-compatible shims keep old internal paths working:
`hbs.qc.bc` → `hbs.qc.beltrami`, `hbs.utils.tool_functions` → `hbs.utils.cast` + `hbs.qc.beltrami`.

## Forward solvers (2.0.0)

`get_hbs` takes `method="angular"` (default since 2.0.0) or `method="zipper"`.

| | `method="angular"` | `method="zipper"` (1.1.0 behavior) |
|---|---|---|
| Return `cw` | `None` | `ConformalWelding` |
| Mesh | fixed `(density=0.04, circle=512)`; `circle_point_num`/`density` ignored | built from `circle_point_num`/`density` |
| Pass `disk=` | used as-is, no mesh fallback | used as-is |
| Input smoothing | failure-only ladder (see below) | failure-only Gaussian retry inside welding |
| Symmetric / sharp shapes | handles them without λ-normalization | λ-normalization may stall and return the GHBS representative |

Both keep the same return contract `(hbs[face], field, cw_or_None, disk)` feeding
`reconstruct_from_hbs` unchanged.

### Failure-only input ladder (angular)

The angular solver needs a smooth boundary. Pixel-staircase contours from
`cv2.findContours` and noisy outlines make the raw harmonic-measure density go
negative and the solve fails outright. In that case — and only then — the input
is retried as:

```
raw → arclength resample(500) → resample(500)+Gaussian σ=2 → resample(500)+Gaussian σ=4
```

The first rung that passes the reconstruction legality gate is returned, and the
result is flagged:

```python
from hbs.angular import angular_mu_from_boundary
mu, field, disk, result, stats = angular_mu_from_boundary(bound)
stats["boundary_rung"]     # "raw" | "resample500" | "resample500+s2" | "resample500+s4"
stats["input_smoothed"]    # True iff a fallback rung was used
```

A `UserWarning` is emitted whenever a fallback rung is used: rescued shapes are
usable but measurably degraded, so they must not be treated as equivalent to a
raw-input solve. If every rung fails, `RuntimeError` is raised (with the last
underlying error chained) instead of returning a degenerate curve.

Measured on the frozen `shape100-v1` set, both engines run through the shipped
entry points and scored with the same evaluator (legal = reconstruction is a
simple closed curve):

| | `angular` (default) | `zipper` |
|---|---|---|
| legal end-to-end | **90/100** | 91/100 (5 more solve but self-intersect) |
| median wall clock | **0.67 s** | 2.24 s |
| median Chamfer, 86 common | **3.84e-05** | 4.84e-04 |
| median excess over sampling floor | **1.42×** | 25.8× |

Angular wins the Chamfer comparison on 81 of those 86 cases (p = 9.6e-19).
Measured separately on real MPEG-7 `cv2` contours (similarity- and
orientation-invariant shape fidelity < 0.30 counted as usable):

| sample | raw angular | + ladder | Zipper |
|---|---|---|---|
| 95 cases, design probe | 68 | **72** | 69 |
| 55 cases, re-checked through the packaged wheel | 37 | **39** | 38 |

On cases both engines solve, angular fidelity is far lower (median 0.0019 vs
0.1371). Rescued rungs land at fidelity 0.049–0.157 — usable but degraded, hence
the warning. The remaining outlines (22/95 in the probe, 16/55 in the recheck)
are rejected by both engines: negatively oriented harmonic-measure density with
no smooth reparameterization that fixes it.

## Usage Example

Compute HBS from image



```python
from hbs.utils.boundary import get_boundary
from hbs import get_hbs

# Boundary points extraction
# 
# we don't actually restrict the method of boundary extraction,
# but this package provides a simple implementation as follows
img_path = 'img/example.png'
bound_point_num = 250
bound = get_boundary(img_path, bound_point_num)

# HBS computing
#
# it requires boundary points arranged in clockwise order as input, in `np.complex`
#
# 2.0.0: the default method="angular" ignores circle_point_num/density and uses a
# fixed (density=0.04, circle=512) mesh. Pass `disk=` to control the mesh, or
# method="zipper" for the 1.1.0 behavior (where these two arguments do apply).
hbs, hbs_mapping, cw, disk = get_hbs(bound)
```

Reconstruct shape from HBS
```python
from hbs import reconstruct_from_hbs

## `disk` must be a DiskMesh corresponding to give `hbs`
## (always reuse the `disk` returned by get_hbs)
bound, _, _, _ = reconstruct_from_hbs(hbs, disk)
```

Compute conformal welding from image

```python
from hbs.utils.boundary import get_boundary
from hbs.conformal_welding import get_conformal_welding

# Boundary points extraction
img_path = 'img/example.png'
bound_point_num = 250
bound = get_boundary(img_path, bound_point_num)

# Conformal welding computing
# 
# it also requires boundary points arranged in clockwise order as input, in `np.complex`
cw = get_conformal_welding(bound)
```

Please also refer to `example.ipynb`

## Dependencies

- NumPy
- SciPy

Optional (extras):

- `hbs[plot]` → Matplotlib (for `hbs.plotting`)
- `hbs[boundary]` → OpenCV (for `hbs.utils.boundary`)
- `hbs[all]` → both

Core compute (HBS, mesh, LSQC, welding) works without matplotlib.

## License

MIT — see [`LICENSE`](LICENSE).

## Contributing

Contributions are welcome via pull requests.

## Code Author
-  Chenran Lin
