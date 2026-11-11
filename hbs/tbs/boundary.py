"""Circle boundary inputs for the fixed-trace P1 solver.

Two representations are provided:

* :class:`FourierCircleTrace` - a finite Fourier angle lift, kept because the
  Fejer construction and the acceptance diagnostics take this form;
* :class:`LocalC2Boundary` - the exact-rational local C2 rounding of a
  monotone welding table. This is the representation ``solve_tbs`` consumes.

This module carries geometry, not evidence. Interval enclosures, derivative
certificates and outward-rounded transfer budgets live in a separate
certification stack outside this package and are deliberately not duplicated
here: a candidate map produced by this package is a local finite-P1 object
that still needs that independent certification.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction as Q
import math

import numpy as np


def _float_upper(value):
    result = float(value)
    if Q(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _atan_bounds(q, terms):
    total = sum(((-1)**k * q**(2*k+1) / Q(2*k+1) for k in range(terms)), Q(0))
    next_term = (-1)**terms * q**(2*terms+1) / Q(2*terms+1)
    return min(total, total + next_term), max(total, total + next_term)


def _pi_upper():
    """Machin-formula upper bound for pi, bit-identical to ``pi_interval().hi``.

    The local C2 rounding widths depend on this bound. Reproducing the same
    rational enclosure keeps the widths -- and therefore the trace -- exactly
    equal to the ones used by the certification stack.
    """
    a, b = _atan_bounds(Q(1, 5), 48), _atan_bounds(Q(1, 239), 16)
    return _float_upper(16*a[1] - 4*b[0])


_PI_UPPER = _pi_upper()


@dataclass(frozen=True)
class FourierCircleTrace:
    """Complete smooth trace psi = theta + offset + sum a sin(n theta) + b cos(n theta).

    The triangle-inequality derivative lower bound must be strictly positive.
    These budgets are ordinary floating point diagnostics, not outward
    enclosures.
    """

    terms: tuple = ()
    offset: float = 0.0

    def __post_init__(self):
        for n, a, b in self.terms:
            if int(n) != n or n < 1 or not np.isfinite([a, b]).all():
                raise ValueError("positive integer frequencies and finite amplitudes required")
        if not np.isfinite(self.offset) or self.derivative_lower <= 0:
            raise ValueError("a strictly positive analytic derivative lower bound is required")

    @property
    def derivative_lower(self):
        return 1 - sum(n*np.hypot(a, b) for n, a, b in self.terms)

    def angle(self, theta):
        theta = np.asarray(theta, float)
        return theta + self.offset + sum(
            (a*np.sin(n*theta) + b*np.cos(n*theta) for n, a, b in self.terms),
            np.zeros_like(theta),
        )

    def derivative(self, theta):
        theta = np.asarray(theta, float)
        return 1 + sum(
            (n*(a*np.cos(n*theta) - b*np.sin(n*theta)) for n, a, b in self.terms),
            np.zeros_like(theta),
        )

    def record(self):
        return dict(representation="complete finite Fourier angle lift",
                    terms=[list(term) for term in self.terms], offset=self.offset,
                    analytic_derivative_lower_float=self.derivative_lower)


def _poly_bounds(coeff):
    degree = len(coeff) - 1
    bernstein = [sum((coeff[j]*Q(math.comb(k, j), math.comb(degree, j))
                      for j in range(k+1)), Q(0))
                 for k in range(degree+1)]
    return min(bernstein), max(bernstein)


class LocalC2Boundary:
    """Faithful finite C2 rounding of a piecewise-linear welding lift.

    Only disjoint symmetric neighborhoods of knots change. The derivative is a
    convex cubic smoothstep between its two positive adjacent slopes. Values,
    first and second derivatives match exactly in rational normalized
    coordinates. Period is exactly one.
    """

    def __init__(self, beta, alpha, *, angle_error_limit=1e-4):
        beta, alpha = np.asarray(beta, float), np.asarray(alpha, float)
        tau = Q(float(2*np.pi))
        if (beta.ndim != 1 or alpha.shape != beta.shape or len(beta) < 4
                or beta[0] != 0 or alpha[0] != 0):
            raise ValueError("paired zero-start monotone angle tables required")
        x = [Q(float(v))/tau for v in beta] + [Q(1)]
        y = [Q(float(v))/tau for v in alpha] + [Q(1)]
        if any(x[i] >= x[i+1] or y[i] >= y[i+1] for i in range(len(beta))):
            raise ValueError("strictly increasing degree-one table required")
        slopes = [(y[i+1]-y[i])/(x[i+1]-x[i]) for i in range(len(beta))]
        tolerance = Q(angle_error_limit)/(2*Q(float(_PI_UPPER)))
        widths = []
        for j in range(len(beta)):
            left = x[j] - x[j-1] if j else 1 - x[-2]
            right = x[j+1] - x[j]
            width = min(left, right)/8
            delta = abs(slopes[j] - slopes[j-1])
            if delta:
                width = min(width, 16*tolerance/(3*delta))
            widths.append(width)
        self._build(x, y, widths)

    def _build(self, x, y, widths):
        if (len(x) != len(y) or len(widths) != len(x)-1 or x[0] != 0 or y[0] != 0
                or x[-1] != 1 or y[-1] != 1
                or any(x[i] >= x[i+1] or y[i] >= y[i+1] for i in range(len(widths)))
                or any(w <= 0 for w in widths)):
            raise ValueError("valid rational degree-one data and positive widths required")
        self.knots, self.values, self.widths = x, y, widths
        count = len(widths)
        slopes = [(y[j+1]-y[j])/(x[j+1]-x[j]) for j in range(count)]
        self.slopes = slopes
        self.is_rotation = all(a == b for a, b in zip(x, y))
        segments = []
        for j in range(count):
            w = widths[j]
            ql, qr = slopes[j-1], slopes[j]
            delta = qr - ql
            start = x[j] - w
            length = 2*w
            uc = [y[j] - w*ql - start, length*(ql-1), Q(0), 2*w*delta, -w*delta]
            qc = [ql, Q(0), 3*delta, -2*delta]
            pc = [Q(0), 3*delta/w, -3*delta/w]
            qb = (min(ql, qr), max(ql, qr))
            pb = (min(Q(0), 3*delta/(4*w)), max(Q(0), 3*delta/(4*w)))
            segments.append((start, start+length, uc, qc, pc, (_poly_bounds(uc), qb, pb)))
            end = x[j+1] - widths[(j+1) % count]
            start = x[j] + w
            length = end - start
            if length <= 0:
                raise ArithmeticError("rounding neighborhoods overlap")
            uc = [y[j] + w*qr - start, length*(qr-1)]
            segments.append((start, end, uc, [qr], [Q(0)],
                             (_poly_bounds(uc), (qr, qr), (Q(0), Q(0)))))
        self.segments = segments
        self.domain_start = segments[0][0]
        self.starts_float = np.array([float(s[0]) for s in segments])
        self.starts_exact = [s[0] for s in segments]
        self.lengths_float = np.array([float(s[1]-s[0]) for s in segments])

    def _numeric(self, t):
        t = np.asarray(t, float)
        shape = t.shape
        x = t.ravel()
        shift = np.floor(x - float(self.domain_start))
        x = x - shift
        index = np.searchsorted(self.starts_float, x, side='right') - 1
        index = np.clip(index, 0, len(self.segments)-1)
        local = (x - self.starts_float[index])/self.lengths_float[index]
        uncertain = ((self.lengths_float[index] < 256*np.finfo(float).eps)
                     | (local < 0) | (local > 1)
                     | (abs(x - self.starts_float[index]) < 4*np.finfo(float).eps)
                     | (abs(x - (self.starts_float[index]
                                 + self.lengths_float[index])) < 4*np.finfo(float).eps))
        for k in np.flatnonzero(uncertain):
            point = Q(float(t.ravel()[k]))
            point -= math.floor(point - self.domain_start)
            j = max(0, min(len(self.segments)-1, bisect_right(self.starts_exact, point)-1))
            index[k] = j
            local[k] = float((point - self.segments[j][0])
                             / (self.segments[j][1] - self.segments[j][0]))
        result = np.zeros((3, len(x)))
        for i in np.unique(index):
            mask = index == i
            s = local[mask]
            segment = self.segments[i]
            result[0, mask] = np.polynomial.polynomial.polyval(
                s, [float(c) for c in segment[2]])
            if len(segment[3]) == 1:
                result[1, mask] = float(segment[3][0])
                result[2, mask] = 0
            else:
                # Positive convex weights avoid catastrophic q_left-q_right
                # cancellation. This evaluates the same polynomial.
                qleft = segment[3][0]
                qright = sum(segment[3], Q(0))
                result[1, mask] = ((1-s)**2*(1+2*s)*float(qleft)
                                   + s*s*(3-2*s)*float(qright))
                result[2, mask] = float(6*(qright-qleft)/(segment[1]-segment[0]))*s*(1-s)
        return tuple(v.reshape(shape) for v in result)

    def angle(self, theta):
        theta = np.asarray(theta, float)
        return theta + 2*np.pi*self._numeric(theta/(2*np.pi))[0]

    def derivative(self, theta):
        return self._numeric(np.asarray(theta)/(2*np.pi))[1]

    def record(self):
        return dict(kind='local_c2_welding',
                    representation='exact rational local C2 rounding of normalized monotone welding table',
                    knots=[str(x) for x in self.knots],
                    values=[str(y) for y in self.values],
                    widths=[str(w) for w in self.widths], offset=0.)

    @classmethod
    def from_record(cls, record):
        value = cls.__new__(cls)
        value._build(list(map(Q, record['knots'])),
                     list(map(Q, record['values'])),
                     list(map(Q, record['widths'])))
        return value
