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
- `_pipeline.py`: The single backend after the welding — orient/anchor/resample,
  FEM harmonic extension, `μ = H_z̄/H_z`, source-circle rotation gauge
- `_engines.py`: The two forward engines behind one registry
  (`angular`, `zipper`); they never fall back to each other
- `angular.py`: Angular-velocity forward solver (the default since 2.0.0)
- `_angular/`: Vendored angular-velocity engine
  (`direct.py`, `joint.py`); depends only on NumPy/SciPy
- `conformal_welding.py`: Conformal welding algorithm implementation
- `extension.py`: P1 finite-element harmonic extension (shared by both engines)
- `cache.py`: On-disk cache for the disk mesh and its FEM operator
- `gauges.py`: Source-circle rotation gauges — I₂ single anchor and the paper's
  M3+M4 closed form, with the `auto` policy that picks between them
- `weld_compare.py`: Welding agreement metric (degree-1 circle homeomorphism,
  source-rotation aligned)
- `mesh.py`: Mesh generation and processing
- `plotting.py`: Optional plotting helpers (matplotlib is NOT required for compute)
- `utils/`:
  - `boundary.py`: Boundary processing tools (requires opencv-python)
  - `cast.py`: Complex ↔ real array conversion
  - `eval.py`: Similarity-invariant shape fidelity (`fidelity_sim`)
  - `geodesic_welding.py`: Geodesic welding
  - `mobius.py`: Möbius transformations
  - `poisson.py`: Dense Poisson integral — **experiment reference only**, library
    code uses `extension.fem_extend`
  - `zipper.py`: Zipper algorithm implementation
- `qc/`: Quasiconformal mapping algorithms
  - `beltrami.py`: Beltrami coefficient computation + mu_chop
  - `lsqc.py`: Least squares quasiconformal mapping algorithm

Top-level `import hbs` exposes the full public API:
`get_hbs`, `get_hbs_result`, `reconstruct_from_hbs`, `HBSResult`,
`get_conformal_welding`, `ConformalWelding`, `compare_engines`, `EngineAgreement`,
`compare_welding`, `compare_conformal_welding`, `welding_curve`,
`WeldingCurve`, `WeldingAgreement`,
`select_gauge`, `GaugeChoice`, `GAUGES`, `NORMALIZATIONS`, `DEFAULT_NORMALIZE`,
`get_beltrami_coefficient`, `lsqc_solver`, `mu_chop`,
`get_unit_disk`, `get_rect`, `get_unit_disk_in_rect`, `Mesh`, `DiskMesh`.

See [`CHANGELOG.md`](https://github.com/ChanceAroundYou/hbs_python/blob/main/CHANGELOG.md)
for the solver comparison, measured benchmarks, and everything that changed in
2.0.0 / 2.0.1 / 2.0.2 / 2.1.0 / 2.2.0 / 2.3.0 / 2.3.1.

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
# boundary points may be given in either orientation (normalized internally).
# `method` selects the engine, and the two are independent implementations that
# never fall back to each other:
#   "angular" (default) — boundary-layer logarithmic single-layer potentials
#   "zipper"            — Bishop/Marshall zipper
# `normalize` selects the source-circle rotation gauge:
#   "auto" (default) — I₂ when |I₂|/Σ|term| ≥ 1e-2, else the paper's M3+M4
#   "i2"             — single anchor; one equation, one unknown, no branch, but
#                      it is ~1500x more sensitive to mesh discretization than M₃
#   "m34"            — paper (5.8)+(5.9) closed form, two branches
#   "none"           — raw engine output
# `get_hbs_result` additionally returns `gauge` (which one ran, both
# confidences, and the M3+M4 branch margin).
hbs, hbs_mapping, cw, disk = get_hbs(bound, method="angular")
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
# same engine registry as get_hbs, so (x, y) is bitwise identical to the welding
# get_hbs used internally for the same method.
cw = get_conformal_welding(bound, method="angular")
```

Inspect which engine ran and whether a degraded resample was used:

```python
from hbs import get_hbs_result

result = get_hbs_result(bound, method="zipper")
result.engine   # "zipper"
result.rung     # "raw", or e.g. "resample500+s2" for a degraded fallback
```

Compare the two engines on the same shape (removes the source-circle rotation
first — doing this by hand gives misleading numbers):

```python
from hbs import compare_engines, get_hbs_result
import hbs.cache as cache

disk, fem = cache.load_unit_disk(0.01, 500)
z = get_hbs_result(bound, method="zipper",  normalize="none", disk=disk)
a = get_hbs_result(bound, method="angular", normalize="none", disk=disk)
agreement = compare_engines(z.welding, a.welding, disk, fem, 500)
agreement.welding_rms, agreement.mu_rms
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

MIT — see [`LICENSE`](https://github.com/ChanceAroundYou/hbs_python/blob/main/LICENSE).

## Contributing

Contributions are welcome via pull requests.

## Code Author
-  Chenran Lin
