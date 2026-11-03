"""三维固定迹求解器的单元测试（小网格，只验证契约与单调性）。"""
from __future__ import annotations

import numpy as np
import pytest

from hbs.volumetric import (
    SpheroidChart,
    ball_tetra_mesh,
    distortion_metrics,
    solve_fixed_trace,
    solve_softmax,
    solve_softmax_continuation,
)


@pytest.fixture(scope="module")
def case():
    # 与 2026-11-03 迁移基线同一网格：初值 max_K ≈ 7.46，6 次迭代后 ≈ 2.2。
    chart = SpheroidChart(a=1.0, c=1.5)
    mesh = ball_tetra_mesh(boundary_count=48, interior_count=10, seed=0)
    initial = chart.cone_initial(mesh.points)
    mesh.trace = chart.boundary(mesh.points[mesh.interior_count :])
    return chart, mesh, initial


def test_solve_fixed_trace_keeps_the_trace_and_reduces_max_k(case):
    chart, mesh, initial = case
    before = distortion_metrics(initial, mesh)
    result = solve_fixed_trace(chart, mesh, maxiter=6)
    assert result["boundary_residual"] <= 1e-12
    after = result["final"]
    assert after["negative_or_zero_tets"] == 0
    assert after["min_J"] > 0
    assert after["max_K"] < before["max_K"]
    assert np.isfinite(result["epigraph_K"])
    np.testing.assert_array_equal(result["values"][mesh.interior_count :], mesh.trace)


def test_solve_fixed_trace_is_deterministic(case):
    chart, mesh, _ = case
    first = solve_fixed_trace(chart, mesh, maxiter=3)
    second = solve_fixed_trace(chart, mesh, maxiter=3)
    np.testing.assert_array_equal(first["values"], second["values"])


def test_solve_fixed_trace_rejects_a_different_trace(case):
    chart, mesh, initial = case
    bad = initial.copy()
    bad[mesh.interior_count :] += 1e-3
    with pytest.raises(ValueError, match="boundary trace"):
        solve_fixed_trace(chart, mesh, initial_values=bad)


def test_solve_fixed_trace_accepts_box_bounds(case):
    chart, mesh, initial = case
    result = solve_fixed_trace(chart, mesh, maxiter=4, bounds_delta=0.5)
    assert result["boundary_residual"] <= 1e-12
    assert result["final"]["negative_or_zero_tets"] == 0
    shift = np.abs(
        result["values"][: mesh.interior_count] - initial[: mesh.interior_count]
    )
    assert np.all(shift <= 0.5 + 1e-12)


def test_solve_fixed_trace_rejects_nonpositive_bounds_delta(case):
    chart, mesh, _ = case
    with pytest.raises(ValueError, match="bounds_delta"):
        solve_fixed_trace(chart, mesh, maxiter=1, bounds_delta=0.0)


def test_solve_softmax_keeps_the_trace_and_reduces_max_k(case):
    _, mesh, initial = case
    before = distortion_metrics(initial, mesh)
    result = solve_softmax(mesh, initial, beta=40.0, maxiter=30, delta=0.02)
    np.testing.assert_array_equal(result["values"][mesh.interior_count :], mesh.trace)
    assert result["metrics"]["max_K"] < before["max_K"]
    assert result["metrics"]["negative_or_zero_tets"] == 0
    assert result["evaluations"] >= 1


def test_solve_softmax_requires_a_trace():
    mesh = ball_tetra_mesh(boundary_count=12, interior_count=3, seed=0)
    with pytest.raises(ValueError, match="trace"):
        solve_softmax(mesh, mesh.points.copy())


def test_solve_softmax_rejects_wrong_initial_shape(case):
    _, mesh, initial = case
    with pytest.raises(ValueError, match="shape"):
        solve_softmax(mesh, initial[:5])


def test_solve_softmax_rejects_a_different_trace(case):
    _, mesh, initial = case
    bad = initial.copy()
    bad[mesh.interior_count :] += 1e-3
    with pytest.raises(ValueError, match="boundary trace"):
        solve_softmax(mesh, bad)


def test_softmax_continuation_records_each_stage(case):
    _, mesh, initial = case
    out = solve_softmax_continuation(
        mesh, initial, betas=(10, 20), maxiter=10, delta=0.02
    )
    assert [item["beta"] for item in out["history"]] == [10.0, 20.0]
    assert out["metrics"]["negative_or_zero_tets"] == 0
    assert out["seconds"] >= 0.0
