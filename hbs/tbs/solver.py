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

The optional ``initialization="lse"`` path globalizes the start with a
log-sum-exp surrogate and keeps every stage only if it is feasible and does
not raise the exact peak. It is a different candidate than the plain Tutte
start: the value it returns is not bit-identical, so it is off by default and
each use has to be recorded against its own baseline.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix, csr_matrix, hstack

from .boundary import LocalC2Boundary
from .geometry import (disk_mesh, face_fields, face_gradients, face_state,
                       regularized_target_boundary, tutte_initial_map)

# Beta ladder for the log-sum-exp cold start. The recorded probes put the last
# useful gain at 1e6: beyond that the exact peak moved by less than 1e-6 while
# the stage cost kept growing.
LSE_BETAS = (16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576)
LSE_ITERATIONS = 200
LSE_MAX_BISECTIONS = 30


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


def _lse_context(z, faces, gx, gy, n_interior, values):
    """Cached quantities for the smooth-surrogate stages."""
    fz, fbar, _ = face_fields(values, faces, gx, gy)
    return {"z": np.asarray(z), "faces": np.asarray(faces),
            "gx": gx, "gy": gy, "n_interior": int(n_interior),
            "values": np.asarray(values, dtype=complex),
            "ratio_scales": np.maximum(np.abs(fz)**2 + np.abs(fbar)**2, 1e-12)}


def _lse_evaluate(x, context):
    """Exact face moduli, their gradients in the interior coordinates, min J."""
    n_interior = context["n_interior"]
    values = context["values"].copy()
    values[:n_interior] = x[:n_interior] + 1j*x[n_interior:2*n_interior]
    ratios, jacobian = face_state(values, context["faces"], context["gx"],
                                  context["gy"])
    _, constraint_jacobian = _face_constraints(
        np.r_[x[:n_interior], x[n_interior:2*n_interior], 0.0],
        values_template=values, faces=context["faces"], gx=context["gx"],
        gy=context["gy"], n_interior=n_interior,
        ratio_scales=context["ratio_scales"])
    return ratios, (-constraint_jacobian[:, :-1]).tocsr(), float(np.min(jacobian))


def _softmax(beta, ratios):
    """Softmax weights of ``beta*r_f`` and the smooth maximum ``F_beta``."""
    shifted = beta*(ratios - np.max(ratios))
    weights = np.exp(shifted)
    total = float(weights.sum())
    return weights/total, float(np.max(ratios) + np.log(total)/beta)


def _lse_stage(x, context, beta, max_iterations):
    """One L-BFGS stage on ``F_beta = log(sum exp(beta*r_f))/beta``.

    ``F_beta`` is an upper surrogate for ``max_f r_f``; the stage never
    reports the surrogate as a result, only the point it reaches.
    """
    worst = {"trial": np.inf, "accepted": np.inf}

    def objective(point):
        ratios, gradients, minimum_jacobian = _lse_evaluate(point, context)
        weights, value = _softmax(beta, ratios)
        worst["trial"] = min(worst["trial"], minimum_jacobian)
        return value, np.asarray(gradients.T @ weights).ravel()

    def track(point):
        _, _, minimum_jacobian = _lse_evaluate(point, context)
        worst["accepted"] = min(worst["accepted"], minimum_jacobian)

    started = perf_counter()
    result = minimize(objective, x, jac=True, method="L-BFGS-B", callback=track,
                      options={"maxiter": max_iterations,
                               "maxfun": 20*max_iterations, "ftol": 0.0,
                               "gtol": 1e-12, "maxcor": 20})
    ratios, _, minimum_jacobian = _lse_evaluate(result.x, context)
    weighted, value = _softmax(beta, ratios)
    return result.x, {
        "beta": float(beta), "seconds": float(perf_counter() - started),
        "iterations": int(result.nit), "evaluations": int(result.nfev),
        "exact_peak": float(np.max(ratios)),
        "surrogate_value": value,
        "surrogate_gap": float(value - np.max(ratios)),
        "minimum_jacobian": float(minimum_jacobian),
        "worst_min_jacobian_accepted": float(worst["accepted"]),
        "worst_min_jacobian_trial": float(worst["trial"]),
        "effective_active_faces": float(1.0/np.sum(weighted**2)),
    }


def _lse_initial_map(z, faces, gx, gy, n_interior, start, *, betas, iterations,
                     max_bisections=LSE_MAX_BISECTIONS):
    """Smooth-surrogate cold start: beta ladder with a monotone feasible filter.

    Every stage is checked at its endpoint before it is kept: the point must be
    a feasible P1 map and must not raise the exact peak. A stage that fails is
    retracted along the segment from the previous accepted point by halving,
    and dropped if even the smallest fraction fails. Without that filter a
    finite-beta stage can walk off a converged minimax point (recorded on the
    archived all-face start), because the surrogate's minimizer is not the
    nonsmooth stationary point.

    Within a stage the optimizer visits trial points that may fold; only the
    accepted iterates and the stage endpoints are constrained here.
    """
    context = _lse_context(z, faces, gx, gy, n_interior, start)
    x = np.r_[start[:n_interior].real, start[:n_interior].imag]
    ratios, _, minimum_jacobian = _lse_evaluate(x, context)
    accepted_peak = float(np.max(ratios))
    if not np.isfinite(accepted_peak) or minimum_jacobian <= 0.0:
        raise ValueError("log-sum-exp start must be a feasible P1 QC map")
    history = []
    total = 0.0
    for beta in betas:
        candidate, info = _lse_stage(x, context, beta, iterations)
        total += info["seconds"]
        fraction, endpoint = 1.0, None
        for _ in range(max_bisections + 1):
            trial = x + fraction*(candidate - x)
            trial_ratios, _, trial_jacobian = _lse_evaluate(trial, context)
            trial_peak = float(np.max(trial_ratios))
            feasible = bool(np.isfinite(trial_peak) and trial_jacobian > 0.0
                            and trial_peak < 1.0)
            if feasible and trial_peak <= accepted_peak*(1.0 + 1e-14):
                endpoint = (trial_peak, float(trial_jacobian))
                break
            fraction *= 0.5
        info["stage_seconds"] = info["seconds"]
        info["accepted_fraction"] = fraction
        info["accepted_peak"] = endpoint[0] if endpoint else accepted_peak
        info["stage_retained"] = endpoint is not None
        if endpoint is None:
            info["minimum_jacobian_at_endpoint"] = minimum_jacobian
        else:
            x, accepted_peak = x + fraction*(candidate - x), endpoint[0]
            minimum_jacobian = endpoint[1]
            info["minimum_jacobian_at_endpoint"] = endpoint[1]
        history.append(info)
    values = start.copy()
    values[:n_interior] = x[:n_interior] + 1j*x[n_interior:2*n_interior]
    return values, {"betas": [float(beta) for beta in betas],
                    "iterations": int(iterations),
                    "stages": history,
                    "final_peak": accepted_peak,
                    "final_minimum_jacobian": float(minimum_jacobian),
                    "total_seconds": total,
                    "stages_retained": int(sum(row["stage_retained"]
                                               for row in history))}


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
                 max_backtracks, armijo, relative_stationarity, peak_window,
                 peak_tolerance, feasibility_tolerance, lp_method):
    started = perf_counter()
    if peak_window < 1:
        raise ValueError("peak_window must be at least 1")
    if not np.isfinite(peak_tolerance) or peak_tolerance < 0.0:
        raise ValueError("peak_tolerance must be finite and non-negative")
    values = np.asarray(initial_values, dtype=complex).copy()
    trace = np.asarray(boundary_values, dtype=complex)
    if values.shape != z.shape or np.max(np.abs(values[n_interior:] - trace)) > 1e-12:
        raise ValueError("start must preserve the frozen boundary trace")
    gx, gy, _ = face_gradients(z, faces)
    ratios, jacobian = face_state(values, faces, gx, gy)
    if np.min(jacobian) <= 0.0 or np.max(ratios) >= 1.0:
        raise ValueError("start is not a feasible P1 QC map")
    start_k = float(np.max(ratios))
    accepted_peaks = [start_k]
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

        # Uniform progress gate: the model-based ratio above decays far too
        # slowly to fire in any practical budget, so the observable objective
        # gets its own rule -- stop once the peak fell by less than
        # ``peak_tolerance`` (relative) across ``peak_window`` accepted steps.
        accepted_peaks.append(accepted_record["k_after"])
        if peak_tolerance > 0.0 and len(accepted_peaks) > peak_window:
            earlier = accepted_peaks[-peak_window - 1]
            relative_decrease = (earlier - accepted_peaks[-1])/earlier
            if relative_decrease <= peak_tolerance:
                record["peak_window_relative_decrease"] = relative_decrease
                stop_reason = "relative_peak_change_below_threshold"
                break

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
              peak_window=25, peak_tolerance=0.0,
              feasibility_tolerance=1e-10, lp_method="highs-ipm",
              initialization="tutte", lse_betas=None,
              lse_iterations=LSE_ITERATIONS):
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
    initialization, lse_betas, lse_iterations
        ``"tutte"`` (default) starts the all-face LP steps at the Tutte
        embedding, exactly as before. ``"lse"`` first runs the smooth
        surrogate ``F_beta = log(sum_f exp(beta*r_f))/beta`` along a beta
        ladder and hands its endpoint to the LP steps. Each ladder stage is
        kept only if its endpoint is a feasible P1 map and does not raise the
        exact peak, otherwise the stage is retracted by bisection; the exact
        peak therefore never increases during the cold start. The surrogate is
        a globalization aid, not a minimax solver: the reported ``final_k`` is
        always the exact ``max_f r_f`` of the returned map. ``lse_betas``
        defaults to :data:`LSE_BETAS`; ``lse_iterations`` is the L-BFGS budget
        per stage.
    peak_window, peak_tolerance
        Uniform peak-progress stop rule. Once ``peak_window`` accepted steps
        have accumulated, stop when the peak modulus fell by less than
        ``peak_tolerance`` (relative) over that window, and report
        ``stop_reason="relative_peak_change_below_threshold"``.
        ``peak_tolerance=0`` (the default) disables the rule and leaves the
        step budget in charge.

    Returns
    -------
    dict
        ``values`` (complex mapped vertices), ``z``/``faces``/``n_interior``/
        ``theta`` (the mesh actually used), ``start_k``/``final_k`` (peak
        ``|mu|``), ``boundary_polygon`` diagnostics, per-step ``history``,
        ``initialization`` (the cold-start record, including the LSE stages
        when used) and the solver ``stop_reason``.

    Notes
    -----
    Local finite-P1 candidate only. No global, continuum or mesh-limit claim,
    and no certificate: independent upper/lower bounds are produced by the
    certification stack, not here. Sub-ULP continuation in exact rational
    arithmetic is likewise a separate research tool; this solver stops at the
    float64 trust-radius floor and reports why.

    ``stop_reason`` is one of ``iteration_limit``, ``linear_subproblem_failure``,
    ``relative_model_descent_below_threshold``,
    ``trust_radius_below_coordinate_roundoff_floor``,
    ``trust_radius_below_minimum_after_rejections`` or
    ``relative_peak_change_below_threshold``. The model-descent ratio decays
    by only about two orders over hundreds of steps, so in practice the
    default 1e-5 gate almost never fires; the peak-window rule measures the
    objective itself and is the one to use when a stated accuracy, rather
    than a step budget, should end the solve. Either way the tolerance is an
    accuracy choice, not a stationarity certificate.
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

    if initialization not in ("tutte", "lse"):
        raise ValueError("initialization must be 'tutte' or 'lse'")
    lse_record = None
    cold_start_seconds = 0.0
    if initialization == "lse":
        betas = LSE_BETAS if lse_betas is None else tuple(float(b) for b in lse_betas)
        if not betas or min(betas) <= 0.0:
            raise ValueError("lse_betas must be a non-empty sequence of positives")
        gx, gy, _ = face_gradients(z, faces)
        started = perf_counter()
        initial, lse_record = _lse_initial_map(
            z, faces, gx, gy, n_interior, initial, betas=betas,
            iterations=int(lse_iterations))
        cold_start_seconds = perf_counter() - started
        lse_record["seconds_including_setup"] = cold_start_seconds

    values, result = _minimize_p1(
        z, faces, boundary_polygon, initial, n_interior=n_interior,
        max_steps=max_steps, initial_radius=initial_radius,
        minimum_radius=minimum_radius, maximum_radius=maximum_radius,
        max_backtracks=max_backtracks, armijo=armijo,
        relative_stationarity=relative_stationarity,
        peak_window=peak_window, peak_tolerance=peak_tolerance,
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
            "peak_window": peak_window, "peak_tolerance": peak_tolerance,
            "feasibility_tolerance": feasibility_tolerance,
            "lp_method": lp_method,
            "initialization": initialization,
            "lse_betas": None if lse_record is None else lse_record["betas"],
            "lse_iterations": int(lse_iterations)},
        "initialization": {"method": initialization,
                           "seconds": cold_start_seconds, "lse": lse_record},
        "cold_start_seconds": cold_start_seconds,
        # ``wall_seconds`` keeps its original meaning (the all-face LP phase
        # only) so recorded baselines stay comparable; the cold start is timed
        # separately above.
        "total_wall_seconds": result["wall_seconds"] + cold_start_seconds,
        "scope": ("local finite-P1 minimax candidate on a frozen complete-circle "
                  "trace; no global, continuum or mesh-limit claim; no certificate"),
    })
    return result
