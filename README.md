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
- `angular.py`: Angular-velocity forward solver (the default since 2.0.0)
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

See [`CHANGELOG.md`](CHANGELOG.md) for the solver comparison, measured
benchmarks, and everything that changed in 2.0.0 / 2.0.1.

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
# `method` picks the forward solver; see CHANGELOG.md for the difference.
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
