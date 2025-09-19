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

Or install from source:
1. Clone this repository:
   ```bash
   git clone https://github.com/ChanceAroundYou/hbs_python.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Main Modules

- `hbs.py`: HBS algorithm for computing HBS and recontructing shape from HBS
- `conformal_welding.py`: Conformal welding algorithm implementation
- `mesh.py`: Mesh generation and processing
- `utils/`:
  - `boundary.py`: Boundary processing tools
  - `geodesic_welding.py`: Geodesic welding
  - `mobius.py`: Möbius transformations
  - `poisson.py`: Poisson integral implementation
  - `tool_functions.py`: Utility functions
  - `zipper.py`: Zipper algorithm implementation
- `qc/`: Quasiconformal mapping algorithms
  - `bc.py`: Beltrami coefficient computation
  - `lsqc.py`: Least squares quasiconformal mapping algorithm

## Usage Example

Compute HBS from image

```python
from hbs.utils.boundary import get_boundary
from hbs import get_hbs

img_path = 'img/example.jpg'
circle_point_num = 1000
density = 0.01
bound_point_num = 250
bound = get_boundary(img_path, bound_point_num)

hbs, hbs_mapping, cw, disk = get_hbs(bound, circle_point_num, density)
```

Reconstruct shape from HBS
```python
from hbs import reconstruct_from_hbs

## `disk` must be a DiskMesh corresponding to give `hbs`
bound, _, _, _ = reconstruct_from_hbs(hbs, disk)
```

Compute conformal welding from image

```python
from hbs.utils.boundary import get_boundary
from hbs.conformal_welding import get_conformal_welding

img_path = 'img/example.jpg'
bound_point_num = 250
bound = get_boundary(img_path, bound_point_num)

cw = get_conformal_welding(bound)
```

Please also refer to `example.ipynb`

## Dependencies

- NumPy
- SciPy
- Matplotlib
- Opencv

## Contributing

Contributions are welcome via pull requests.

## Code Author
-  Chenran Lin
