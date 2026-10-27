"""Joint full-field least squares for H, with unknown sliding boundary.

The optimizer accepts only an internal conformal coordinate, the physical
curve, and a source mesh. It never accepts beta, an exterior density, a welding
table, reference H, or a converged boundary selector. Internal H nodes and the
physical boundary parameters share one Gauss--Newton system and line search.
"""
from dataclasses import dataclass
from time import perf_counter
import weakref

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline, LinearNDInterpolator
from scipy.optimize import root
from scipy.linalg import lstsq as dense_lstsq
from scipy.sparse import bmat, csr_matrix, diags, hstack, vstack, eye
from scipy.sparse.linalg import splu


_SOURCE_DISK_OPERATOR_CACHE = {}


def _source_disk_operator(mesh, ii, bd):
    """Return exact mesh-only Laplace blocks and their reusable LU factor."""
    key=id(mesh);entry=_SOURCE_DISK_OPERATOR_CACHE.get(key)
    if entry is not None and entry[0]() is mesh:
        return entry[1],entry[2],entry[3],True
    area=abs(mesh.area)
    stiffness=(mesh.Dx.T@diags(area)@mesh.Dx+mesh.Dy.T@diags(area)@mesh.Dy).tocsr()
    mass=np.bincount(mesh.face.ravel(),weights=np.repeat(area/3,3),minlength=len(mesh.vert))
    weak_laplacian=diags(1/np.sqrt(np.pi*mass[ii]))@stiffness[ii]
    A=weak_laplacian[:,ii].tocsr();B=weak_laplacian[:,bd].tocsr()
    factor=splu(A.tocsc())
    try:
        reference=weakref.ref(mesh,lambda _ref,k=key:_SOURCE_DISK_OPERATOR_CACHE.pop(k,None))
    except TypeError:
        return A,B,factor,False
    _SOURCE_DISK_OPERATOR_CACHE[key]=(reference,A,B,factor)
    return A,B,factor,False


class InternalDiskCoordinate:
    """Q on the physical boundary, from the internal metric u=log|Q'| only."""
    def __init__(self, shape, parameter, u, cache_label='internal metric'):
        tic = perf_counter()
        self.shape = shape
        self.count = len(parameter)
        self.dt = 2*np.pi/self.count
        self.u = CubicSpline(np.r_[parameter, 2*np.pi], np.r_[u, u[0]], bc_type='periodic')
        x, w = leggauss(8)
        self.gx, self.gw = (x+1)/2, w/2
        left = np.arange(self.count)*self.dt
        nodes = left[:, None]+self.dt*self.gx
        density = np.exp(self.u(nodes))*abs(shape.derivative(nodes))
        increments = self.dt*(density@self.gw)
        self.total = float(increments.sum())
        self.factor = 2*np.pi/self.total
        self.cumulative = np.r_[0., np.cumsum(increments)]*self.factor
        self.center = 0j
        samples = self.trace(shape.sample_t)[0]
        def eq(v):
            a = complex(*v)
            mean = np.mean((samples-a)/(1-np.conj(a)*samples))
            return [mean.real, mean.imag]
        fit = root(eq, [0., 0.], tol=1e-11)
        self.center = complex(*fit.x)
        if abs(self.center) >= .999 or np.linalg.norm(eq(fit.x)) > 1e-8:
            raise RuntimeError('Internal Q sample gauge failed')
        self.diagnostics = dict(cache=cache_label, read_fields=['parameter', 'u'],
                                setup_seconds=perf_counter()-tic, raw_phase_total=self.total,
                                normalization_multiplier=self.factor,
                                center_real=self.center.real, center_imag=self.center.imag,
                                sample_moment=float(abs(self.trace(shape.sample_t)[0].mean())),
                                log_metric_min=float(np.min(u)), log_metric_max=float(np.max(u)),
                                exterior_fields_read=False)

    def trace(self, parameter):
        phase, speed = self.phase_speed(parameter)
        h = np.exp(1j*phase)
        return h, 1j*h*speed

    def phase_speed(self, parameter):
        p = np.mod(np.asarray(parameter, float), 2*np.pi)
        index = np.minimum((p/self.dt).astype(int), self.count-1)
        left = index*self.dt
        length = p-left
        nodes = left[..., None]+length[..., None]*self.gx
        integral = length*np.sum(np.exp(self.u(nodes))*abs(self.shape.derivative(nodes))*self.gw, axis=-1)
        alpha = self.cumulative[index]+self.factor*integral
        z = np.exp(1j*alpha)
        a = self.center
        speed = self.factor*np.exp(self.u(p))*abs(self.shape.derivative(p))
        speed *= (1-abs(a)**2)/abs(z-a)**2
        phase = alpha+2*np.angle(1-a*np.exp(-1j*alpha))
        return phase, speed


def periodic_lift(theta, p, target, derivative=0):
    spline = CubicSpline(np.r_[theta, 2*np.pi], np.r_[p-theta, p[0]-theta[0]], bc_type='periodic')
    return ((np.asarray(target) if derivative == 0 else 1. if derivative == 1 else 0.)
            + spline(np.mod(target, 2*np.pi), derivative))


def minimum_lift_speed(theta, p):
    cs = CubicSpline(np.r_[theta, 2*np.pi], np.r_[p-theta, p[0]-theta[0]], bc_type='periodic')
    a, b, c = cs.c[:3]
    width = np.diff(cs.x)
    ends = np.minimum(1+c, 1+3*a*width**2+2*b*width+c)
    with np.errstate(divide='ignore', invalid='ignore'):
        critical = -b/(3*a)
    active = (critical > 0) & (critical < width) & np.isfinite(critical)
    values = ends.copy()
    values[active] = np.minimum(values[active], 1+c[active]+2*b[active]*critical[active]+3*a[active]*critical[active]**2)
    return float(values.min())


@dataclass
class JointState:
    z: np.ndarray
    values: np.ndarray
    theta: np.ndarray
    parameter: np.ndarray
    diagnostics: dict


class JointHBSProblem:
    def __init__(self, shape, coordinate, mesh, spectral_weight=1., oversample=4):
        self.shape, self.coordinate, self.mesh = shape, coordinate, mesh
        self.z = mesh.vert[:, 0]+1j*mesh.vert[:, 1]
        bd = np.flatnonzero(abs(abs(self.z)-1) < 1e-10)
        angles = np.mod(np.angle(self.z[bd]), 2*np.pi)
        angles[abs(angles-2*np.pi) < 1e-10] = 0.
        order = np.argsort(angles)
        self.bd, self.theta = bd[order], angles[order]
        self.ii = np.setdiff1d(np.arange(len(self.z)), self.bd)
        self.ni, self.nb = len(self.ii), len(self.bd)
        if np.max(abs(self.theta-2*np.pi*np.arange(self.nb)/self.nb)) > 1e-8:
            raise ValueError('Expected a uniform source-circle mesh')
        # These matrices depend only on the source disk mesh, not the target
        # curve, metric, boundary parameter, or H iterate. Reuse them across
        # continuation stages without carrying any target-specific state.
        self.A,self.B,self.source_factor,self.source_operator_cache_hit = \
            _source_disk_operator(mesh,self.ii,self.bd)
        self.dense_theta = 2*np.pi*np.arange(oversample*self.nb)/(oversample*self.nb)
        basis = np.vstack([np.eye(self.nb), np.eye(self.nb)[0]])
        self.interpolation = CubicSpline(np.r_[self.theta, 2*np.pi], basis, axis=0,
                                        bc_type='periodic')(self.dense_theta)
        self.positive = np.arange(2, len(self.dense_theta)//2)
        t = 2*np.pi*np.arange(8192)/8192
        g, gp = shape.curve(t), shape.derivative(t)
        self.radius_squared = float(np.mean(np.imag(np.conj(g)*gp)))
        if self.radius_squared <= 0:
            raise ValueError('Physical boundary must be CCW')
        self.scale = np.sqrt(self.radius_squared)
        self.weights = np.sqrt(spectral_weight*self.positive)/self.scale

    def initial(self, seed=0, previous=None):
        rng = np.random.default_rng(seed)
        if previous is None:
            p = self.theta.copy()
            for k in range(1, 4):
                p += .003*rng.normal()*np.sin(k*self.theta)/k
            candidate = periodic_lift(self.theta, p, np.mod(np.angle(self.z[self.ii]), 2*np.pi))
            h = self.coordinate.trace(candidate)[0]
            center = self.coordinate.trace(p)[0].mean()
            r = abs(self.z[self.ii])
            interior = (1-r)*center+r*h
            label = 'geometric parameter plus radial candidate fill; no reference data'
        else:
            p = periodic_lift(previous.theta, previous.parameter, self.theta)
            if len(previous.z) == len(self.z) and np.max(abs(previous.z-self.z)) < 1e-12:
                interior = previous.values[self.ii].copy()
            else:
                interp = LinearNDInterpolator(np.c_[previous.z.real, previous.z.imag], previous.values)
                interior = interp(self.z[self.ii].real, self.z[self.ii].imag)
                if not np.isfinite(interior).all():
                    raise RuntimeError('Full-field continuation interpolation left source mesh')
            label = 'previous joint full-field iterate; internal Q updated; no reference data'
        return np.r_[interior.real, interior.imag, p], label

    def unpack(self, x):
        p = x[2*self.ni:]
        h, hp = self.coordinate.trace(p)
        v = np.empty(len(self.z), complex)
        v[self.ii] = x[:self.ni]+1j*x[self.ni:2*self.ni]
        v[self.bd] = h
        return v, p, hp

    def evaluate(self, x, jacobian=False, full_jacobian=True):
        values, p, hp = self.unpack(x)
        lap = self.A@values[self.ii]+self.B@values[self.bd]
        pd = self.dense_theta+self.interpolation@(p-self.theta)
        spectrum = np.fft.fft(self.shape.curve(pd))/len(pd)
        forbidden = spectrum[self.positive]*self.weights
        residual = np.r_[lap.real, lap.imag, forbidden.real, forbidden.imag,
                         spectrum[1].imag/self.scale]
        result = dict(residual=residual, cost=float(residual@residual),
                      harmonic_residual=float(np.linalg.norm(lap)),
                      spectral_residual=float(np.linalg.norm(forbidden)),
                      leading=spectrum[1], values=values, parameter=p,
                      interior_radius_max=float(abs(values[self.ii]).max()))
        if jacobian:
            db = self.B@diags(hp)
            transformed = np.fft.fft(self.shape.derivative(pd)[:, None]*self.interpolation, axis=0)/len(pd)
            blocked = transformed[self.positive]*self.weights[:, None]
            jboundary = np.vstack([blocked.real, blocked.imag, transformed[1].imag[None, :]/self.scale])
            result['boundary_jacobian'] = jboundary
            result['lap_boundary_jacobian'] = db
            if full_jacobian:
                jlap = bmat([[self.A, None, db.real], [None, self.A, db.imag]], format='csr')
                bottom = hstack([csr_matrix((len(jboundary), 2*self.ni)),
                                 csr_matrix(jboundary)], format='csr')
                result['jacobian'] = vstack([jlap, bottom], format='csc')
                pre_lap = bmat([[eye(self.ni), None, db.real],
                               [None, eye(self.ni), db.imag]], format='csr')
                result['preconditioned_jacobian'] = vstack([pre_lap, bottom], format='csc')
        return result

    def preconditioned_normal(self, state):
        """Exact J.T J, using dense BLAS for the spectral border."""
        br, bi = state['lap_boundary_jacobian'].real, state['lap_boundary_jacobian'].imag
        c = state['boundary_jacobian']
        if isinstance(state['lap_boundary_jacobian'], np.ndarray):
            pp = br.T@br+bi.T@bi+c.T@c
            br_block, bi_block = csr_matrix(br), csr_matrix(bi)
        else:
            pp = (br.T@br+bi.T@bi).toarray()+c.T@c
            br_block, bi_block = br, bi
        normal = bmat([[eye(self.ni), None, br_block], [None, eye(self.ni), bi_block],
                       [br_block.T, bi_block.T, csr_matrix(pp)],], format='csc')
        r = state['residual']
        boundary_gradient = (br.T@r[:self.ni]+bi.T@r[self.ni:2*self.ni]
                             + c.T@r[2*self.ni:])
        gradient = np.r_[r[:2*self.ni], boundary_gradient]
        return normal, gradient

    def parameter_speed(self, parameter):
        return minimum_lift_speed(self.theta, parameter)

    def evaluate_candidate(self, x):
        """Evaluate a line-search candidate, with optional model-specific reuse."""
        if self.parameter_speed(x[2*self.ni:]) <= 1e-7:
            return None
        return self.evaluate(x)

    def evaluation_parameter_speed(self, evaluation):
        """Return the admissibility speed associated with an evaluated state."""
        return self.parameter_speed(evaluation['parameter'])

    def final_audit(self, final):
        dense_theta = 2*np.pi*np.arange(8*self.nb)/(8*self.nb)
        pd = periodic_lift(self.theta, final['parameter'], dense_theta)
        sp = np.fft.fft(self.shape.curve(pd))/len(pd)
        kk = np.arange(2, len(pd)//2)
        return float(np.sqrt(np.sum(kk*abs(sp[kk])**2))/self.scale), {}

    def make_state(self, final, diagnostics):
        return JointState(self.z, final['values'], self.theta, final['parameter'], diagnostics)


def solve_joint_hbs(problem, seed=0, previous=None, max_iterations=45, tolerance=1e-7,
                    step_strategy='model_backtracking',training_spectral_tolerance=None,
                    spectral_backend_fallback=None):
    """Solve the joint problem.

    ``backtracking`` is the reference implementation. ``trust_ratio`` tests
    one full step using its actual/predicted reduction ratio. ``model_backtracking``
    chooses a first fraction by minimizing the exact reduced linearized
    residual model, then retains safeguarded geometric backtracking and the
    original actual-cost acceptance rule. These options change globalization,
    not the HBS objective.
    """
    if step_strategy not in ('backtracking', 'trust_ratio', 'model_backtracking'):
        raise ValueError("step_strategy must be 'backtracking', 'trust_ratio', or 'model_backtracking'")
    if spectral_backend_fallback not in (None, 'nufft', 'quadrature'):
        raise ValueError("spectral_backend_fallback must be None, 'nufft', or 'quadrature'")
    start = perf_counter()
    x, initialization = problem.initial(seed, previous)
    trace, damping, reason = [], 1e-9, 'iteration_limit'
    candidate_evaluations = 0
    accepted_evaluation = None
    backend_switches = []
    last_backend_switch_iteration = None
    for iteration in range(max_iterations):
        deferred_jacobian = (spectral_backend_fallback is not None
                             and problem.coefficient_method == 'fft'
                             and hasattr(problem, 'jacobian_from_evaluation'))
        if accepted_evaluation is not None and hasattr(problem, 'jacobian_from_evaluation'):
            state = accepted_evaluation
            accepted_evaluation = None
            has_jacobian = False
        else:
            state = problem.evaluate(x, not deferred_jacobian, full_jacobian=False)
            has_jacobian = not deferred_jacobian
        trace.append(dict(iteration=iteration, cost=state['cost'],
                          harmonic_residual=state['harmonic_residual'],
                          spectral_residual=state['spectral_residual'],
                          min_parameter_speed=problem.evaluation_parameter_speed(state),
                          interior_radius_max=state['interior_radius_max']))

        # Near an FFT solution, compare its spectral residual with the same
        # physical-parameter Gauss sum used by the independent audit. If the
        # FFT residual has reached its apparent floor while the physical sum
        # has not, switch at this iterate and continue the same joint solve.
        # The expensive FFT Jacobian is deliberately deferred until after this
        # check, and every candidate/final endpoint gate remains unchanged.
        near_training_stop = (training_spectral_tolerance is not None
                              and state['harmonic_residual'] < 1e-7
                              and np.linalg.norm(state['residual'][2*problem.ni:])
                                  < training_spectral_tolerance)
        if deferred_jacobian and near_training_stop:
            physical_residual = problem.physical_spectral_residual(
                state, method=spectral_backend_fallback)
            trace[-1].update(physical_spectral_residual=physical_residual)
            if physical_residual > max(8e-5, 10*max(1e-8, state['spectral_residual'])):
                old_method = problem.coefficient_method
                problem.coefficient_method = spectral_backend_fallback
                accepted_evaluation = None
                damping = 1e-9
                last_backend_switch_iteration = iteration
                backend_switches.append(dict(iteration=iteration, from_method=old_method,
                    to_method=spectral_backend_fallback,
                    fft_spectral_residual=float(state['spectral_residual']),
                    physical_spectral_residual=float(physical_residual)))
                state = problem.evaluate(x, False)
                trace[-1].update(cost=state['cost'],
                    harmonic_residual=state['harmonic_residual'],
                    spectral_residual=state['spectral_residual'],
                    min_parameter_speed=problem.evaluation_parameter_speed(state),
                    interior_radius_max=state['interior_radius_max'],
                    switched_spectral_backend=spectral_backend_fallback)
                has_jacobian = False
        if np.sqrt(state['cost']) < tolerance:
            reason = 'residual_tolerance'
            break
        if (training_spectral_tolerance is not None
                and state['harmonic_residual']<1e-7
                and np.linalg.norm(state['residual'][2*problem.ni:])<training_spectral_tolerance):
            # The independently refined final audit still decides acceptance.
            # Do not spend dozens of iterations oversolving an FFT surrogate
            # whose discretization floor exceeds the requested training fit.
            reason='training_components_tolerance'
            break
        if not has_jacobian:
            state = problem.jacobian_from_evaluation(state)
        # Invertible coordinate change delta_y=A*delta_H for internal DOFs.
        # This is applied within each joint linearized step, never after a
        # separate converged boundary solve.
        border = state['boundary_jacobian']
        boundary_residual = state['residual'][2*problem.ni:]
        scale = 1/np.maximum(np.linalg.norm(border, axis=0), 1e-12)
        scaled_border = border*scale[None, :]
        lap = state['residual'][:problem.ni]+1j*state['residual'][problem.ni:2*problem.ni]
        accepted = False
        for attempt in range(5):
            try:
                # Exact block elimination of ONE joint linearized LS problem.
                # Do not square the large Laplace/boundary scale ratio.
                augmented = np.vstack([scaled_border, np.sqrt(damping)*np.eye(len(scale))])
                rhs = np.r_[-boundary_residual, np.zeros(len(scale))]
                delta_boundary = dense_lstsq(augmented, rhs, cond=1e-12, lapack_driver='gelsy')[0]*scale
                delta_y = -lap-state['lap_boundary_jacobian']@delta_boundary
                delta_internal = (problem.source_factor.solve(delta_y.real)
                                  +1j*problem.source_factor.solve(delta_y.imag))
                step = np.r_[delta_internal.real, delta_internal.imag, delta_boundary]
            except (RuntimeError, ValueError):
                damping = max(1e-7, 10*damping)
                continue
            if step_strategy == 'trust_ratio':
                fractions = (1.,)
            elif step_strategy == 'model_backtracking':
                # The eliminated interior block cancels the linearized
                # harmonic residual exactly. Minimize the remaining quadratic
                # model along this same joint step to choose the first trial;
                # retain geometric backtracking and the original actual-cost
                # acceptance test as safeguards against nonlinear mismatch.
                harmonic_cost = float(np.vdot(lap, lap).real)
                boundary_direction = border@delta_boundary
                curvature = harmonic_cost+float(boundary_direction@boundary_direction)
                slope = -harmonic_cost+float(boundary_residual@boundary_direction)
                predicted_fraction = (-slope/curvature if curvature > 0 else 1.)
                predicted_fraction = float(np.clip(predicted_fraction, 2.**-12, 1.))
                # Try geometrically smaller fractions of the model's best
                # linearized step. Every candidate still passes the original
                # nonlinear objective, positivity, and interior feasibility gates.
                fractions = (predicted_fraction*2.**-power for power in range(9))
            else:
                fractions = (2.**-power for power in range(17))
            for fraction in fractions:
                candidate = x+fraction*step
                candidate_evaluations += 1
                trial = problem.evaluate_candidate(candidate)
                if trial is None:
                    if step_strategy == 'trust_ratio':
                        damping = min(1e8, max(1e-7, 10*damping))
                        break
                    continue
                feasible = (trial['leading'].real > 0
                            and trial['interior_radius_max'] <= 1+1e-8)
                if step_strategy == 'trust_ratio':
                    predicted_residual = boundary_residual+border@delta_boundary
                    predicted_reduction = state['cost']-float(predicted_residual@predicted_residual)
                    actual_reduction = state['cost']-trial['cost']
                    ratio = (actual_reduction/predicted_reduction
                             if predicted_reduction > 0 else -np.inf)
                    accepted_trial = feasible and actual_reduction > 0 and ratio > 1e-4
                else:
                    ratio = None
                    accepted_trial = (feasible
                                      and trial['cost'] < state['cost']*(1-1e-5*fraction))
                if accepted_trial:
                    x = candidate
                    accepted_evaluation = trial
                    trace[-1].update(step_fraction=fraction,
                                     interior_step_norm=float(np.linalg.norm(fraction*step[:2*problem.ni])),
                                     boundary_step_norm=float(np.linalg.norm(fraction*step[2*problem.ni:])),
                                     **({'reduction_ratio': float(ratio)} if ratio is not None else {}))
                    if step_strategy == 'trust_ratio':
                        damping = (max(1e-12, damping/3) if ratio > .75
                                   else min(1e8, max(1e-12, damping*3)) if ratio < .25
                                   else damping)
                    else:
                        damping = (min(1e4, max(1e-8, 10*damping)) if fraction < .25
                                   else max(1e-12, damping/3) if fraction == 1 else damping)
                    accepted = True
                    break
                if step_strategy == 'trust_ratio':
                    damping = min(1e8, max(1e-7, 10*damping))
                    break
            if accepted:
                break
            damping = max(1e-7, 10*damping)
        if not accepted:
            reason = 'line_search_stalled'
            break
        if np.linalg.norm(step) < 1e-10*(1+np.linalg.norm(x)):
            reason = 'step_tolerance'
            break
    final = problem.evaluate(x)
    dense_residual, extra = problem.final_audit(final)
    gp, gc = problem.mesh.Dz@final['values'], problem.mesh.Dc@final['values']
    mu = gc/gp
    area = abs(problem.mesh.area)
    face = problem.mesh.face
    e1 = final['values'][face[:, 1]]-final['values'][face[:, 0]]
    e2 = final['values'][face[:, 2]]-final['values'][face[:, 0]]
    src1 = problem.z[face[:, 1]]-problem.z[face[:, 0]]
    src2 = problem.z[face[:, 2]]-problem.z[face[:, 0]]
    jac = np.imag(np.conj(e1)*e2)/np.imag(np.conj(src1)*src2)
    post_switch_iterations = (None if last_backend_switch_iteration is None
                              else len(trace)-last_backend_switch_iteration)
    diagnostics = dict(seconds=perf_counter()-start, initialization=initialization,
                       training_spectral_tolerance=training_spectral_tolerance,
                       step_strategy=step_strategy, candidate_evaluations=candidate_evaluations,
                       iterations=len(trace),
                       iterations_after_backend_switch=post_switch_iterations,
                       continuation_iterations=(len(trace) if post_switch_iterations is None
                                                else post_switch_iterations),
                       stop_reason=reason, objective=final['cost'],
                       harmonic_residual=final['harmonic_residual'],
                       spectral_residual=final['spectral_residual'], dense_spectral_residual=dense_residual,
                       minimum_parameter_speed=problem.evaluation_parameter_speed(final),
                       interior_radius_max=final['interior_radius_max'],
                       min_jacobian=float(jac.min()), nonpositive_faces=int(np.sum(jac <= 0)),
                       nonpositive_area_fraction=float(area[jac <= 0].sum()/area.sum()),
                       max_abs_mu=float(abs(mu).max()),
                       circle_nodes=problem.nb, interior_nodes=problem.ni,
                       source_operator_cache_hit=bool(problem.source_operator_cache_hit),
                       spectral_backend_switches=backend_switches,
                       no_boundary_first=True, reference_data_received=False,
                       linear_solver='one joint Gauss-Newton step; exact block elimination and border QR; simultaneous full-field update',
                       converged=bool(final['harmonic_residual'] < 1e-6 and dense_residual < 2e-4),
                       trace=trace, **extra)
    return problem.make_state(final, diagnostics)
