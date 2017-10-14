import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
from scipy.interpolate import PPoly
from .closedcurve import ClosedCurve
from ._compat import *


class Splinep(ClosedCurve):
    def __init__(self, xk, yk):
        assert(len(xk) == len(yk))
        self._xk = np.asarray(xk)
        self._yk = np.asarray(yk)
        if abs(self._xk[0] - self._xk[-1]) > np.spacing(1):
            self._xk = np.hstack([self._xk, self._xk[0]])
            self._yk = np.hstack([self._yk, self._yk[0]])

        ppArray, chordalArcLength = makeSpline(xk, yk)

        def position(t):
            t = np.asarray(t).reshape(-1)
            t = chordalArcLength * t
            zre = self.ppArray[(0, 0)](t)
            zim = self.ppArray[(1, 0)](t)
            return zre + 1j * zim

        def tangent(t):
            t = np.asarray(t).reshape(-1)
            t = chordalArcLength * t
            zre = self.ppArray[(0, 1)](t)
            zim = self.ppArray[(1, 1)](t)
            return chordalArcLength * (zre + 1j * zim)

        super(Splinep, self).__init__(positionfun=position,
                                      tangentfun=tangent,
                                      bounds=(0.0, 1.0))

        self.ppArray = ppArray
        self.chordalArcLength = chordalArcLength

    @classmethod
    def from_complex_list(cls, lst):
        xk = [item.real for item in lst]
        yk = [item.imag for item in lst]
        xk = np.asarray(xk)
        yk = np.asarray(yk)
        return Splinep(xk, yk)

    @classmethod
    def from_two_vectors(cls, xs, ys):
        return Splinep(xs, ys)

    @property
    def xpts(self):
        return list(self._xk)

    @property
    def ypts(self):
        return list(self._yk)

    @property
    def zpts(self):
        return self._xk + 1j*self._yk

    def clone(self):
        return Splinep(self._xk, self._yk)

    def apply(self, op):
        raise NotImplemented('todo')

    def arclength(self):
        return self.chordalArcLength

    def __call__(self, t):
        return self.point(t)

    def second(self, t):
        t = np.asarray(t).reshape(-1)
        t = self.modparam(t) * self.arclength()
        zre = self.ppArray[(0, 2)](t)
        zim = self.ppArray[(1, 2)](t)
        zs = zre + 1j * zim
        return self.arclength()**2 * zs

    def __str__(self):
        fh = StringIO()
        fh.write('splinep object:\n\n')
        fh.write('  defined with %d spline knots,\n' % len(self._xk))
        fh.write('  total chordal arc length %s\n\n' % self.arclength())
        return fh.getvalue()

    def __add__(self, scalar):
        xs = [x + scalar.real for x in self._xk]
        ys = [y + scalar.imag for y in self._yk]
        return Splinep(xs, ys)


class PiecewisePolynomial(object):
    def __init__(self, breaks, coefs):
        self.coefs = coefs
        self.breaks = breaks.reshape(1, -1)
        self.__f = PPoly(self.coefs.T, self.breaks[0, :])

    def __call__(self, t):
        return self.__f(t)


def mkpp(breaks, coeffs):
    """Simplfied version of MATLABs mkpp function using scipy
    """
    return PiecewisePolynomial(breaks, coeffs[:, :])


def makeSpline(x, y):
    """This algorithm is from "PERIODIC CUBIC SPLINE INTERPOLATION USING
    PARAMETRIC SPLINES" by W.D. Hoskins and P.R. King, Algorithm 73, The
    Computer Journal, 15, 3(1972) P282-283. Fits a parametric periodic
    cubic spline through n1 points (x(i), y(i)) (i = 1, ... ,n1) with
    x(1) = x(n1) and y(1) = y(n1). This function returns the first three
    derivatives of x and y, the chordal distances h(i) of (x(i),y(i)) and
    (x(i + 1), y(i + 1)) (i = 1, ..., n1 - 1) with h(n1) = h(1) and the
    total distance.

    Thomas K. DeLillo, Lianju Wang 07-05-99.
    modified a bit by E. Kropf, 2013, 2014.
    ported to Python by A. Walker, 2015
    """
    x = np.asarray(x, dtype=np.double)
    y = np.asarray(y, dtype=np.double)

    if (abs(x[0] - x[-1]) > np.spacing(1)) or (abs(y[0] - y[-1]) > np.spacing(1)):
        x = np.hstack([x, x[0]])
        y = np.hstack([y, y[0]])

    nk = len(x)
    n = nk - 1
    dx = np.diff(x)
    dy = np.diff(y)
    h = np.sqrt(dx**2 + dy**2)
    tl = np.sum(h)
    h = np.hstack([h, h[0]])
    p = h[:-1]
    q = h[1:]
    a = q / (p + q)
    b = 1 - a

    Amat = np.ones((n, 5))
    Amat[0,   0] = b[-1]
    Amat[:-1, 1] = a[1:]
    Amat[:,   2] *= 2.0
    Amat[0,   3] = 0.0
    Amat[1:,  3] = b[:-1]
    Amat[-1,  4] = a[0]

    data = Amat.T
    diags = np.array([1-n, -1, 0, 1, n-1])
    c = scipy.sparse.spdiags(data, diags, n, n)

    tmp1 = (a * dx / p)
    tmp2 = b * np.hstack([dx[1:], x[1] - x[-1]]) / q
    d1 = 3 * (tmp1 + tmp2)
    x1 = scipy.sparse.linalg.spsolve(c.tocsr(), d1)
    x1 = np.hstack([x1[-1], x1])

    tmp1 = (a * dy / p)
    tmp2 = b * np.hstack([dy[1:], y[1] - y[-1]]) / q
    d2 = 3 * (tmp1 + tmp2)
    y1 = scipy.sparse.linalg.spsolve(c.tocsr(), d2)
    y1 = np.hstack([y1[-1], y1])

    x2 = 2 * (x1[:n] + 2*x1[1:] - 3*dx/p)/p
    y2 = 2 * (y1[:n] + 2*y1[1:] - 3*dy/p)/p
    x2 = np.hstack([x2[-1], x2])
    y2 = np.hstack([y2[-1], y2])

    x3 = np.diff(x2)/p
    y3 = np.diff(y2)/p
    x3 = np.hstack([x3, x3[0]])
    y3 = np.hstack([y3, y3[0]])

    t = np.hstack([0, np.cumsum(h)])

    # Make pp for later evaluation.
    pp = dict()
    pp[(0, 0)] = mkpp(t, np.vstack([x3/6, x2/2, x1, x]).T)
    pp[(0, 1)] = mkpp(t, np.vstack([x3/2, x2, x1]).T)
    pp[(0, 2)] = mkpp(t, np.vstack([x3, x2]).T)

    pp[(1, 0)] = mkpp(t, np.vstack([y3/6, y2/2, y1, y]).T)
    pp[(1, 1)] = mkpp(t, np.vstack([y3/2, y2, y1]).T)
    pp[(1, 2)] = mkpp(t, np.vstack([y3, y2]).T)

    return pp, tl
