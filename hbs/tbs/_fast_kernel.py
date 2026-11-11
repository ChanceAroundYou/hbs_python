"""ctypes wrapper for the fused LSE kernel in ``_fast_kernel.c``.

The C path computes the softmax value and the dense gradient of
``F_beta = log(sum_f exp(beta*r_f))/beta`` in one pass over the faces.  It is a
drop-in replacement for ``solver._lse_evaluate_dense`` + ``_dense_gradient`` +
``_softmax`` when the shared library can be built; callers must keep the Python
path as the reference and fall back when :func:`available` is false.

Build happens on first use into the system temp directory, keyed by the source
hash, so the repository never gains a binary artifact.  ``CC`` overrides the
compiler and ``HBS_FAST_KERNEL=0`` disables the extension entirely.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SOURCE = _HERE / "_fast_kernel.c"
_LIBRARY: ctypes.CDLL | None = None
_FAILED = False
_THREADS = 1


def _build() -> Path | None:
    """Compile the source into the temp cache; return the .so path or None."""
    if not _SOURCE.exists():
        return None
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / "hbs_fast_kernel"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"_fast_kernel_{digest}.so"
    if target.exists():
        return target
    compiler = os.environ.get("CC", "gcc")
    command = [compiler, "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
               "-o", str(target), str(_SOURCE), "-lm"]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return target


def _load() -> ctypes.CDLL | None:
    global _LIBRARY, _FAILED
    if _LIBRARY is not None or _FAILED:
        return _LIBRARY
    if os.environ.get("HBS_FAST_KERNEL", "1") == "0":
        _FAILED = True
        return None
    library = _build()
    if library is None:
        _FAILED = True
        return None
    try:
        loaded = ctypes.CDLL(str(library))
    except OSError:
        _FAILED = True
        return None
    loaded.tbs_lse_objective.restype = ctypes.c_int
    loaded.tbs_lse_objective.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)]
    loaded.tbs_lse_value.restype = ctypes.c_int
    loaded.tbs_lse_value.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    _LIBRARY = loaded
    return _LIBRARY


def available() -> bool:
    """True when the compiled kernel is usable."""
    return _load() is not None


def scratch_size(n_faces: int, n_interior: int) -> int:
    """Work-space length expected by :func:`objective` (5*n_faces)."""
    return 5*n_faces


def _pointer(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _as_float(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def prepare(values) -> tuple[np.ndarray, np.ndarray]:
    """Split the complex template once per stage: (values_re, values_im)."""
    values = np.asarray(values, dtype=np.complex128)
    return _as_float(values.real), _as_float(values.imag)


def objective(x, values, faces, gx, gy, n_interior: int, beta: float,
              ratios: np.ndarray, gradient: np.ndarray,
              scratch: np.ndarray | None = None,
              template: tuple[np.ndarray, np.ndarray] | None = None
              ) -> tuple[float, float]:
    """Fill ``gradient`` with the objective gradient; return (value, min J).

    ``x`` is the 2n interior vector, ``values`` the full template (its boundary
    entries are used verbatim), ``ratios`` and ``gradient`` are outputs that
    must be preallocated.  ``template`` is the cached ``prepare(values)`` pair
    and ``scratch`` a persistent 5*n_faces buffer; passing both avoids the
    per-call copies and the internal malloc.  Returns ``(nan, nan)`` when the
    kernel reports a non-finite point.
    """
    library = _load()
    if library is None:
        raise RuntimeError("fast kernel is not available")
    x = _as_float(x)
    if template is None:
        values_re, values_im = prepare(values)
    else:
        values_re, values_im = template
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    gx = _as_float(gx)
    gy = _as_float(gy)
    ratios = np.ascontiguousarray(ratios, dtype=np.float64)
    gradient = np.ascontiguousarray(gradient, dtype=np.float64)
    if scratch is None:
        scratch = np.empty(scratch_size(len(faces), n_interior),
                           dtype=np.float64)
    minimum = ctypes.c_double(0.0)
    value = ctypes.c_double(0.0)
    status = library.tbs_lse_objective(
        _pointer(x), _pointer(values_re), _pointer(values_im),
        faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        _pointer(gx), _pointer(gy),
        ctypes.c_int64(len(faces)), ctypes.c_int64(n_interior),
        ctypes.c_int64(len(values)), ctypes.c_double(beta),
        _pointer(gradient), _pointer(ratios), ctypes.byref(minimum),
        ctypes.byref(value), _pointer(scratch))
    if status != 0:
        return float("nan"), float("nan")
    return float(value.value), float(minimum.value)


def value(x, values, faces, gx, gy, n_interior: int, beta: float) -> float:
    """Objective value only (allocation-free C path)."""
    library = _load()
    if library is None:
        raise RuntimeError("fast kernel is not available")
    x = _as_float(x)
    values = np.asarray(values, dtype=np.complex128)
    values_re = _as_float(values.real)
    values_im = _as_float(values.imag)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    gx = _as_float(gx)
    gy = _as_float(gy)
    out = ctypes.c_double(0.0)
    minimum = ctypes.c_double(0.0)
    status = library.tbs_lse_value(
        _pointer(x), _pointer(values_re), _pointer(values_im),
        faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        _pointer(gx), _pointer(gy), ctypes.c_int64(len(faces)),
        ctypes.c_int64(n_interior), ctypes.c_double(beta),
        ctypes.byref(out), ctypes.byref(minimum))
    if status != 0:
        return float("nan")
    return float(out.value)