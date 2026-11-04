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


def test_peak_window_gate_fires_at_the_window_and_reports_the_reason():
    beta, alpha = _table(64)
    result = solve_tbs(beta, alpha, boundary_count=64, spacing=0.2,
                       max_steps=40, peak_window=2, peak_tolerance=1.0)
    assert result["stop_reason"] == "relative_peak_change_below_threshold"
    assert result["accepted_steps"] == 2
    assert result["p1_qc_feasible"]
    assert result["boundary_trace_max_difference"] == 0.0
    peaks = [result["start_k"]] + [row["accepted_step"]["k_after"]
                                   for row in result["history"] if row["accepted"]]
    reported = result["history"][-1]["peak_window_relative_decrease"]
    assert np.isclose(reported, (peaks[-3] - peaks[-1])/peaks[-3])


def test_peak_window_gate_is_off_by_default_and_only_fires_when_reached():
    beta, alpha = _table(64)
    default = solve_tbs(beta, alpha, boundary_count=64, spacing=0.2, max_steps=12)
    assert default["parameters"]["peak_tolerance"] == 0.0
    assert default["stop_reason"] == "iteration_limit"
    # A tolerance far below what 12 steps reach must not fire either.
    tight = solve_tbs(beta, alpha, boundary_count=64, spacing=0.2, max_steps=12,
                      peak_tolerance=1e-12)
    assert tight["stop_reason"] == "iteration_limit"
    assert "peak_window_relative_decrease" not in tight["history"][-1]


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


LSE_TEST_BETAS = (8.0, 64.0, 512.0)


def test_lse_cold_start_is_off_by_default():
    beta, alpha = _table(64)
    result = solve_tbs(beta, alpha, boundary_count=64, spacing=0.2, max_steps=4)
    assert result["parameters"]["initialization"] == "tutte"
    assert result["parameters"]["lse_betas"] is None
    assert result["initialization"] == {"method": "tutte", "seconds": 0.0,
                                        "lse": None}
    assert result["cold_start_seconds"] == 0.0
    assert np.isclose(result["total_wall_seconds"], result["wall_seconds"])


def test_lse_cold_start_reaches_a_lower_peak_than_the_tutte_start():
    beta, alpha = _table(64)
    mesh = dict(boundary_count=64, spacing=0.2)
    base = solve_tbs(beta, alpha, max_steps=0, **mesh)
    lse = solve_tbs(beta, alpha, max_steps=0, initialization="lse",
                    lse_betas=LSE_TEST_BETAS, lse_iterations=60, **mesh)
    assert lse["initialization"]["method"] == "lse"
    assert lse["final_k"] < base["start_k"]
    assert lse["p1_qc_feasible"]
    assert lse["final_minimum_jacobian"] > 0.0
    assert lse["boundary_trace_max_difference"] == 0.0
    assert lse["cold_start_seconds"] > 0.0
    assert lse["total_wall_seconds"] >= lse["wall_seconds"]
    stages = lse["initialization"]["lse"]["stages"]
    assert [row["beta"] for row in stages] == list(LSE_TEST_BETAS)
    assert all(np.isfinite(row["exact_peak"]) for row in stages)


def test_lse_stages_never_raise_the_exact_peak():
    """The stage filter is what makes the cold start safe from a warm point."""
    beta, alpha = _table(64)
    mesh = dict(boundary_count=64, spacing=0.2)
    warm = solve_tbs(beta, alpha, max_steps=6, **mesh)["values"]
    restarted = solve_tbs(beta, alpha, max_steps=0, initial_values=warm,
                          initialization="lse", lse_betas=LSE_TEST_BETAS,
                          lse_iterations=60, **mesh)
    assert restarted["final_k"] <= restarted["start_k"]*(1.0 + 1e-14)
    lse = restarted["initialization"]["lse"]
    assert lse["final_peak"] <= restarted["start_k"]*(1.0 + 1e-14)
    assert lse["final_minimum_jacobian"] > 0.0
    for row in lse["stages"]:
        assert 0.0 < row["accepted_fraction"] <= 1.0
        assert row["minimum_jacobian_at_endpoint"] > 0.0


def test_lse_parameters_are_validated():
    beta, alpha = _table(64)
    for kwargs in ({"initialization": "slp"}, {"initialization": "lse",
                                               "lse_betas": ()},
                   {"initialization": "lse", "lse_betas": (0.0, 8.0)},
                   {"initialization": "lse", "lse_betas": (8.0, -1.0)}):
        try:
            solve_tbs(beta, alpha, boundary_count=64, spacing=0.2, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs} must be rejected")
