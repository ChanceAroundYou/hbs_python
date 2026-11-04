"""Fixed-complete-circle welding: local P1 minimax solver.

The problem solved here is the finite-dimensional one. The boundary trace
(the complete circle welding) is frozen; every interior vertex of a P1
triangulation of the disk is free; the objective is the largest facewise
Beltrami modulus

    max_f |mu_f|,   mu_f = f_zbar / f_z

of the piecewise-linear map. Each step linearizes every face modulus, solves
the all-face sparse linear minimax problem inside a box trust region, and
accepts the trial only after reevaluating the exact nonlinear objective and
every face Jacobian. Rejected trials shrink the trust region.

Scope. This returns a local finite-P1 candidate. It is not a global minimax
proof, not a continuum T-map, and carries no certificate: the independent
upper bound (interval enclosures of a composed circle map) and lower bound
(Radau/Schiffer circle integrals) live in the separate certification stack.
Mesh-limit and convergence-in-refinement claims need their own evidence.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack

from .boundary import LocalC2Boundary
from .geometry import (disk_mesh, face_fields, face_gradients, face_state,
                       regularized_target_boundary, tutte_initial_map)


def _face_constraints(x, *, values_template, faces, gx, gy, n_interior,
                      ratio_scales):
    """Return facewise ``kappa-|mu_f| >= 0`` and its analytic Jacobian.

    A 1e-10 relative smoothing avoids the undefined derivative of
    ``|f_zbar|`` at zero. It is many orders below the discretization error.
    The denominator is regularized by the same fixed face scale so that
    degenerate ``f_z=0`` faces remain strongly infeasible when ``kappa<1``.
    """
    values = values_template.copy()
    values[:n_interior] = x[:n_interior] + 1j*x[n_interior:2*n_interior]
    kappa = float(x[-1])
    fz, fbar, _ = face_fields(values, faces, gx, gy)
    fz2, fb2 = np.abs(fz)**2, np.abs(fbar)**2
    eps2 = 1e-20*np.asarray(ratio_scales, dtype=float)
    ratio = np.sqrt(fb2 + eps2)/np.sqrt(fz2 + eps2)
    constraints = kappa - ratio

    n_faces = len(faces)
    cz = 0.5*(gx - 1j*gy)
    cb = 0.5*(gx + 1j*gy)
    rows, cols, data = [], [], []
    for local in range(3):
        da_x, da_y = cz[:, local], 1j*cz[:, local]
        db_x, db_y = cb[:, local], 1j*cb[:, local]
        dr_x = ratio*(np.real(np.conj(fbar)*db_x)/(fb2 + eps2)
                      - np.real(np.conj(fz)*da_x)/(fz2 + eps2))
        dr_y = ratio*(np.real(np.conj(fbar)*db_y)/(fb2 + eps2)
                      - np.real(np.conj(fz)*da_y)/(fz2 + eps2))
        nodes = faces[:, local]
        active = nodes < n_interior
        face_ids = np.flatnonzero(active)
        vertex_ids = nodes[active]
        rows.extend((face_ids, face_ids))
        cols.extend((vertex_ids, n_interior + vertex_ids))
        data.extend((-dr_x[active], -dr_y[active]))
    rows.append(np.arange(n_faces, dtype=np.int64))
    cols.append(np.full(n_faces, 2*n_interior, dtype=np.int64))
    data.append(np.ones(n_faces, dtype=float))
    jacobian = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_faces, 2*n_interior + 1),
    ).tocsr()
    jacobian.sum_duplicates()
    return constraints, jacobian


def _linear_minimax_step(ratios, gradients, radius, *, feasibility_tolerance,
                         lp_method):
    """Solve the all-face LP with u=d/radius and normalized epigraph s=t/radius."""
    peak = float(np.max(ratios))
    n_variables = gradients.shape[1]
    # Every row is retained: a face omitted from a bundle cannot silently
    # become the blocker without appearing in this subproblem.
    matrix = hstack([gradients, -csr_matrix(np.ones((len(ratios), 1)))],
                    format="csr")
    result = linprog(
        np.r_[np.zeros(n_variables), 1.0],
        A_ub=matrix,
        b_ub=(peak - ratios)/radius,
        bounds=[(-1.0, 1.0)]*n_variables + [(None, None)],
        method=lp_method,
        options={"primal_feasibility_tolerance": feasibility_tolerance,
                 "dual_feasibility_tolerance": feasibility_tolerance},
    )
    if not result.success:
        return None, {"success": False, "status": int(result.status),
                      "message": str(result.message),
                      "iterations": int(getattr(result, "nit", 0))}
    direction = radius*np.asarray(result.x[:-1], dtype=float)
    linearized_peak = float(np.max(ratios + gradients @ direction))
    return direction, {
        "success": True,
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "lp_epigraph_change": float(radius*result.fun),
        "normalized_epigraph_change": float(result.fun),
        "predicted_peak_change": linearized_peak - peak,
        "linearized_peak": linearized_peak,
        "maximum_coordinate_step": float(np.max(np.abs(direction))),
    }


def _update_trust_radius(radius, *, maximum_radius, minimum_radius, ratio,
                         step_boundary_fraction, accepted_fraction):
    """Update the trust radius, adapting to the accepted dyadic backtrack."""
    if not 0.0 < accepted_fraction <= 1.0:
        raise ValueError("accepted_fraction must be in (0, 1]")
    if accepted_fraction < 1.0:
        factor = accepted_fraction**0.5
        if ratio < 0.25:
            factor = min(factor, 0.25)
        candidate, action = radius*factor, "geometric_backtrack_shrink"
    elif ratio < 0.25:
        candidate, action = 0.25*radius, "geometric_low_ratio_shrink"
    elif ratio > 0.75 and step_boundary_fraction >= 0.9:
        candidate, action = 2.0*radius, "geometric_expand_boundary_step"
    else:
        candidate, action = radius, "geometric_keep"
    return min(maximum_radius, max(minimum_radius, candidate)), action


def _minimize_p1(z, faces, boundary_values, initial_values, *, n_interior,
                 max_steps, initial_radius, minimum_radius, maximum_radius,
                 max_backtracks, armijo, relative_stationarity,
                 feasibility_tolerance, lp_method):
    started = perf_counter()
    values = np.asarray(initial_values, dtype=complex).copy()
    trace = np.asarray(boundary_values, dtype=complex)
    if values.shape != z.shape or np.max(np.abs(values[n_interior:] - trace)) > 1e-12:
        raise ValueError("start must preserve the frozen boundary trace")
    gx, gy, _ = face_gradients(z, faces)
    ratios, jacobian = face_state(values, faces, gx, gy)
    if np.min(jacobian) <= 0.0 or np.max(ratios) >= 1.0:
        raise ValueError("start is not a feasible P1 QC map")
    start_k = float(np.max(ratios))
    fz, fbar, _ = face_fields(values, faces, gx, gy)
    ratio_scales = np.maximum(np.abs(fz)**2 + np.abs(fbar)**2, 1e-12)
    radius = float(initial_radius)
    records = []
    stop_reason = "iteration_limit"
    rejected_trials = 0

    for iteration in range(1, max_steps + 1):
        ratios, jacobian = face_state(values, faces, gx, gy)
        peak = float(np.max(ratios))
        x = np.r_[values[:n_interior].real, values[:n_interior].imag, peak]
        _, constraint_jacobian = _face_constraints(
            x, values_template=values, faces=faces, gx=gx, gy=gy,
            n_interior=n_interior, ratio_scales=ratio_scales)
        # c_f = kappa - r_f >= 0, hence -dc_f/dx is grad(r_f).
        gradients = (-constraint_jacobian[:, :-1]).tocsr()
        max_row_l1 = float(np.max(np.asarray(np.abs(gradients).sum(axis=1)).ravel()))
        direction, lp_info = _linear_minimax_step(
            ratios, gradients, radius,
            feasibility_tolerance=feasibility_tolerance, lp_method=lp_method)
        if direction is None:
            stop_reason = "linear_subproblem_failure"
            records.append({"iteration": iteration, "peak": peak, "radius": radius,
                            "lp": lp_info, "accepted": False})
            break

        predicted_drop = max(0.0, peak - lp_info["linearized_peak"])
        stationarity_ratio = predicted_drop/max(radius*max_row_l1, 1e-30)
        coordinate_scale = float(np.max(np.abs(values[:n_interior]))) if n_interior else 0.0
        coordinate_roundoff_floor = 256.0*np.finfo(float).eps*max(1.0, coordinate_scale)
        roundoff_limited_radius = radius < coordinate_roundoff_floor
        if stationarity_ratio <= relative_stationarity or roundoff_limited_radius:
            # The linear model has nothing left to offer at this radius. A
            # multi-radius audit over the same all-face LP can still find
            # descent here; it is a research-side diagnostic, not this core.
            stop_reason = ("trust_radius_below_coordinate_roundoff_floor"
                           if roundoff_limited_radius
                           else "relative_model_descent_below_threshold")
            records.append({
                "iteration": iteration, "peak": peak, "radius": radius,
                "lp": lp_info, "predicted_drop": predicted_drop,
                "max_face_gradient_l1": max_row_l1,
                "relative_model_descent": stationarity_ratio, "accepted": False})
            break

        accepted = False
        accepted_record = None
        for backtrack in range(max_backtracks + 1):
            fraction = 0.5**backtrack
            trial = values.copy()
            displacement = fraction*direction
            trial[:n_interior] += (displacement[:n_interior]
                                   + 1j*displacement[n_interior:2*n_interior])
            trial_ratios, trial_jacobian = face_state(trial, faces, gx, gy)
            trial_peak = float(np.max(trial_ratios))
            model_peak = float(np.max(ratios + gradients @ displacement))
            actual_drop = peak - trial_peak
            model_drop = peak - model_peak
            feasible = bool(np.all(np.isfinite(trial_ratios))
                            and np.min(trial_jacobian) > 0.0
                            and trial_peak < 1.0)
            armijo_ok = actual_drop >= armijo*model_drop
            if feasible and model_drop > 0.0 and actual_drop > 0.0 and armijo_ok:
                accepted = True
                accepted_record = {
                    "backtrack": int(backtrack), "fraction": float(fraction),
                    "k_after": trial_peak, "actual_drop": actual_drop,
                    "model_drop_at_fraction": model_drop,
                    "actual_to_model_ratio": actual_drop/model_drop,
                    "minimum_jacobian": float(np.min(trial_jacobian)),
                    "maximum_coordinate_step": float(np.max(np.abs(displacement)))}
                values = trial
                break
            rejected_trials += 1

        record = {"iteration": iteration, "peak_before": peak,
                  "radius_before": radius,
                  "predicted_drop_full_step": predicted_drop,
                  "max_face_gradient_l1": max_row_l1,
                  "relative_model_descent": stationarity_ratio,
                  "lp": lp_info, "accepted": accepted,
                  "accepted_step": accepted_record}
        if not accepted:
            radius *= 0.25
            record["radius_after"] = radius
            records.append(record)
            if radius < minimum_radius:
                stop_reason = "trust_radius_below_minimum_after_rejections"
                break
            continue

        ratio = accepted_record["actual_to_model_ratio"]
        step_boundary_fraction = (accepted_record["maximum_coordinate_step"]
                                  / max(radius, 1e-30))
        accepted_fraction = float(accepted_record["fraction"])
        radius, radius_update_action = _update_trust_radius(
            radius, maximum_radius=maximum_radius, minimum_radius=minimum_radius,
            ratio=ratio, step_boundary_fraction=step_boundary_fraction,
            accepted_fraction=accepted_fraction)
        record.update({"trust_ratio": ratio,
                       "step_boundary_fraction": step_boundary_fraction,
                       "accepted_fraction": accepted_fraction,
                       "radius_update_action": radius_update_action,
                       "radius_after": radius})
        records.append(record)

    final_ratios, final_jacobian = face_state(values, faces, gx, gy)
    final_k = float(np.max(final_ratios))
    return values, {
        "start_k": start_k,
        "final_k": final_k,
        "relative_peak_decrease": (start_k - final_k)/start_k,
        "final_minimum_jacobian": float(np.min(final_jacobian)),
        "boundary_trace_max_difference": float(np.max(np.abs(values[n_interior:] - trace))),
        "p1_qc_feasible": bool(np.min(final_jacobian) > 0.0 and final_k < 1.0),
        "stop_reason": stop_reason,
        "accepted_steps": int(sum(row["accepted"] for row in records)),
        "rejected_trials": int(rejected_trials),
        "final_trust_radius": float(radius),
        "wall_seconds": float(perf_counter() - started),
        "history": records,
    }


def solve_tbs(beta=None, alpha=None, *, trace=None, spacing=0.1, boundary_count=128,
              initial_values=None, mesh=None, max_steps=200,
              initial_radius=1e-4, minimum_radius=1e-12, maximum_radius=1e-3,
              max_backtracks=20, armijo=1e-4, relative_stationarity=1e-5,
              feasibility_tolerance=1e-10, lp_method="highs-ipm"):
    """Minimize ``max_face |mu|`` over P1 self-maps of the disk, trace frozen.

    Parameters
    ----------
    beta, alpha
        Monotone degree-one welding table in radians, both starting at 0 and
        ending below ``2*pi``. Converted to a :class:`LocalC2Boundary`, the
        same exact-rational C2 rounding used by the research pipeline.
    trace
        Alternative boundary object with an ``angle(theta)`` method
        (:class:`LocalC2Boundary` or :class:`FourierCircleTrace`). Mutually
        exclusive with ``beta``/``alpha``.
    spacing, boundary_count
        Uniform disk mesh resolution. ``spacing=0.1, boundary_count=128``
        gives 421 vertices / 800 faces and matches the recorded probes.
    initial_values, mesh
        Optional warm start and ``(z, n_interior, theta, faces)`` mesh. The
        default is a fresh mesh plus a Tutte embedding of the target polygon.

    Returns
    -------
    dict
        ``values`` (complex mapped vertices), ``z``/``faces``/``n_interior``/
        ``theta`` (the mesh actually used), ``start_k``/``final_k`` (peak
        ``|mu|``), ``boundary_polygon`` diagnostics, per-step ``history`` and
        the solver ``stop_reason``.

    Notes
    -----
    Local finite-P1 candidate only. No global, continuum or mesh-limit claim,
    and no certificate: independent upper/lower bounds are produced by the
    certification stack, not here. Sub-ULP continuation in exact rational
    arithmetic is likewise a separate research tool; this solver stops at the
    float64 trust-radius floor and reports why.
    """
    if trace is None and alpha is None and hasattr(beta, "angle"):
        trace, beta = beta, None
    if (beta is None) != (alpha is None):
        raise ValueError("beta and alpha must be supplied together")
    if trace is not None and beta is not None:
        raise ValueError("supply either a welding table or a trace, not both")
    if trace is None:
        if beta is None:
            raise ValueError("a welding table (beta, alpha) or a trace is required")
        trace = LocalC2Boundary(beta, alpha)

    if mesh is not None:
        z, n_interior, theta, faces = mesh
        z = np.asarray(z, dtype=np.complex128)
        faces = np.asarray(faces, dtype=np.int64)
        theta = np.asarray(theta, dtype=float)
    else:
        z, n_interior, theta, faces = disk_mesh(spacing, boundary_count)
    boundary_polygon, polygon_record = regularized_target_boundary(trace, theta)

    if initial_values is None:
        initial = tutte_initial_map(faces, len(z), n_interior, boundary_polygon)
    else:
        initial = np.asarray(initial_values, dtype=complex)
        if initial.shape != z.shape:
            raise ValueError("initial_values must have one entry per mesh vertex")

    values, result = _minimize_p1(
        z, faces, boundary_polygon, initial, n_interior=n_interior,
        max_steps=max_steps, initial_radius=initial_radius,
        minimum_radius=minimum_radius, maximum_radius=maximum_radius,
        max_backtracks=max_backtracks, armijo=armijo,
        relative_stationarity=relative_stationarity,
        feasibility_tolerance=feasibility_tolerance, lp_method=lp_method)

    result.update({
        "values": values,
        "z": z,
        "faces": faces,
        "n_interior": int(n_interior),
        "theta": theta,
        "boundary": trace,
        "boundary_polygon": polygon_record,
        "vertices": int(len(z)),
        "mesh_faces": int(len(faces)),
        "parameters": {
            "spacing": spacing, "boundary_count": boundary_count,
            "max_steps": max_steps,
            "initial_radius": initial_radius, "minimum_radius": minimum_radius,
            "maximum_radius": maximum_radius,
            "max_backtracks": max_backtracks, "armijo": armijo,
            "relative_stationarity": relative_stationarity,
            "feasibility_tolerance": feasibility_tolerance,
            "lp_method": lp_method},
        "scope": ("local finite-P1 minimax candidate on a frozen complete-circle "
                  "trace; no global, continuum or mesh-limit claim; no certificate"),
    })
    return result
