"""Tests for `hbs.tbs`: the fixed-complete-circle TBS solver.

Scope: these check the finite-P1 mechanics (mesh orientation, peak-modulus
descent, the frozen boundary trace, the stop gate). They do not certify the
returned map: the upper interval enclosure, the lower RS bound and the
acceptance replay live outside this package.
"""
import numpy as np

from hbs.tbs import (FourierCircleTrace, LocalC2Boundary, disk_mesh,
                     face_gradients, face_state, solve_tbs)


def _table(count=64):
    beta = 2*np.pi*np.arange(count)/count
    alpha = beta + 0.12*np.sin(2*beta) + 0.03*np.sin(5*beta)
    return beta, alpha


def test_boundary_inputs_are_monotone_and_differentiable():
    beta, alpha = _table(65)
    boundary = LocalC2Boundary(beta, alpha)
    theta = np.linspace(-0.3, 1.3, 2001)
    assert np.all(boundary.derivative(theta) > 0)
    lifted = FourierCircleTrace(((1, 0.2, 0.03), (2, 0.04, -0.02)), 0.3)
    assert np.all(lifted.derivative(np.linspace(0, 2*np.pi, 1001)) > 0)
    assert lifted.record()["terms"] == [[1, 0.2, 0.03], [2, 0.04, -0.02]]


def test_disk_mesh_is_positively_oriented_with_uniform_boundary():
    z, n_interior, theta, faces = disk_mesh(0.2, 32)
    assert len(z) - n_interior == len(theta)
    assert np.array_equal(z[n_interior:], np.exp(1j*theta))
    _, _, areas = face_gradients(z, faces)
    assert np.all(areas > 0)


def test_solve_tbs_decreases_peak_modulus_and_freezes_the_trace():
    beta, alpha = _table(128)
    result = solve_tbs(beta, alpha, boundary_count=128, spacing=0.1, max_steps=8)
    assert result["p1_qc_feasible"]
    assert result["final_k"] < result["start_k"]
    assert result["boundary_trace_max_difference"] == 0.0
    assert result["final_minimum_jacobian"] > 0.0
    assert result["accepted_steps"] >= 1
    ratios, jacobian = face_state(
        result["values"], result["faces"],
        *face_gradients(result["z"], result["faces"])[:2])
    assert np.isclose(np.max(ratios), result["final_k"])
    assert np.min(jacobian) > 0.0


def test_solve_tbs_stops_when_the_linear_model_has_no_relative_descent():
    beta, alpha = _table(64)
    result = solve_tbs(beta, alpha, boundary_count=64, spacing=0.2,
                       max_steps=8, relative_stationarity=1.0)
    assert result["stop_reason"] == "relative_model_descent_below_threshold"
    assert result["accepted_steps"] == 0
    assert result["final_k"] == result["start_k"]
    assert result["p1_qc_feasible"]


def test_solve_tbs_rejects_a_start_that_breaks_the_trace_or_feasibility():
    beta, alpha = _table(64)
    z, n_interior, theta, faces = disk_mesh(0.2, 64)
    broken = np.exp(1j*theta)*(1 + 1e-3)
    start = np.r_[np.zeros(n_interior, dtype=complex), broken]
    try:
        solve_tbs(beta, alpha, mesh=(z, n_interior, theta, faces), initial_values=start)
    except ValueError as error:
        assert "boundary trace" in str(error)
    else:
        raise AssertionError("a start that moves the frozen trace must be rejected")
