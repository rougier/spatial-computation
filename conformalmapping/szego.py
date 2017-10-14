import numpy as np
from numpy.linalg import norm
from .closedcurve import ClosedCurve
from .helpers import *


class SzegoKernel(object):
    def __init__(self, curve, a, opts, **kwargs):
        N = opts.numCollPts

        dt = 1.0 / float(N)
        t = np.arange(0.0, 1.0, dt)
        z = curve.position(t)
        zt = curve.tangent(t)
        zT = zt / np.abs(zt)

        IpA = np.ones((N, N), dtype=np.complex)
        for i in range(1, N):
            cols = np.arange(i)
            zc_zj = z[cols] - z[i]

            tmp1 = np.conjugate(zT[i]/zc_zj)
            tmp2 = zT[cols]/zc_zj
            tmp3 = np.sqrt(np.abs(np.dot(zt[i], zt[cols])))
            tmp4 = (dt/(2.0j*np.pi))

            IpA[i, cols] = (tmp1 - tmp2) * tmp3 * tmp4
            IpA[cols, i] = -np.conjugate(IpA[i, cols])

        y = 1j * np.sqrt(np.abs(zt))/(2*np.pi) * np.conjugate(zT/(z - a))

        # TODO - this is a simplification of the original method
        assert(opts.kernSolMethod in ('auto', 'bs'))
        assert(N < 2048)

        x = np.linalg.solve(IpA, y)

        relresid = norm(y - np.dot(IpA, x)) / norm(y)
        if relresid > 100.0 * np.spacing(1):
            raise Exception('out of tolerance')

        # set output
        self.phiColl = x
        self.dtColl = dt
        self.zPts = z
        self.zTan = zt
        self.zUnitTan = zT


class SzegoOpts(object):
    def __init__(self):
        self.confCenter = 0.0 + 0.0j
        self.numCollPts = 512
        self.kernSolMethod = 'auto'
        self.newtonTol = 10.0 * np.spacing(2.0*np.pi)
        self.trace = False
        self.numFourierPts = 2*256


class Szego(object):
    def __init__(self, curve=None, confCenter=0.0 + 0.0j,
                 opts=None, *args, **kwargs):

        if not isinstance(curve, ClosedCurve):
            raise Exception('Expected a closed curve object')

        self.curve = curve
        self.confCenter = confCenter

        if opts is None:
            opts = SzegoOpts()

        self.numCollPts = opts.numCollPts

        kernel = SzegoKernel(curve, confCenter, SzegoOpts())
        self.phiColl = kernel.phiColl
        self.dtColl = kernel.dtColl
        self.zPts = kernel.zPts
        self.zTan = kernel.zTan
        self.zUnitTan = kernel.zUnitTan

        self.theta0 = np.angle(-1.0j * self.phi(0.0)**2 * self.curve.tangent(0))
        self.Saa = np.sum(np.abs(self.phiColl**2))*self.dtColl
        self.newtTol = opts.newtonTol
        self.beNoisy = opts.trace

    @suppress_warnings
    def kerz_stein(self, ts):
        t = np.asarray(ts).reshape(1, -1)[0, :]
        w = self.curve.position(t)
        wt = self.curve.tangent(t)
        wT = wt / np.abs(wt)

        z = self.zPts
        zt = self.zTan
        zT = self.zUnitTan

        separation = 10 * np.spacing(np.max(np.abs(z)))

        def KS_by_idx(wi, zi):
            # TODO - unflatten this expression and vectorise appropriately when 
            #        futher testing confirms this covers all of the appropriate
            #        cases
            z_w = z[zi] - w[wi]
            tmp1 = wt[wi]*zt[zi]
            tmp2 = np.abs(tmp1)
            tmp3 = np.sqrt(tmp2)
            tmp4 = (2j * np.pi)
            tmp5 = np.conjugate(wT[wi]/z_w)
            tmp6 = zT[zi]/z_w
            tmp7 = tmp5 - tmp6
            out = tmp3 / tmp4 * tmp7
            out[np.abs(z_w) < separation] = 0.0
            return out

        wis = np.arange(len(w))
        zis = np.arange(self.numCollPts)
        A = [KS_by_idx(wi, zis) for wi in wis]
        A = np.vstack(A)
        return A

    def phi(self, ts):
        ts = np.asarray(ts).reshape(1, -1)[0, :]
        v = self.psi(ts) - np.dot(self.kerz_stein(ts), self.phiColl) * self .dtColl
        return v

    def psi(self, ts):
        ts = np.asarray(ts).reshape(1, -1)[0, :]
        wt = self.curve.tangent(ts)
        xs = self.curve.point(ts) - self.confCenter
        tmp1 = np.sqrt(np.abs(wt))
        y = 1.0j / (2*np.pi) / tmp1 * np.conjugate(wt / xs)
        return y

    def theta(self, ts):
        ts = np.asarray(ts).reshape(1, -1)[0, :]
        ph = self.phi(ts)**2
        th = np.angle(-1.0j * ph * self.curve.tangent(ts))
        th = th - self.theta0
        th[ts == 0] = 0
        return th

    def _log(self, msg):
        print(msg)

    @suppress_warnings
    def invtheta(self, s, tol=None):
        assert(np.all(np.diff(s)) > 0)
        assert(np.all(s != 2*np.pi))
        ntol = tol
        if tol is None:
            ntol = self.newtTol

        def f(t, s):
            return s - np.mod(self.theta(t), 2*np.pi)

        t = s / (2 * np.pi)
        assert(t.shape == s.shape)

        btol = 1e-3
        bmaxiter = 20

        nb = np.max([np.ceil(1.0/(2**4 * btol)), np.size(t)])
        if nb > np.size(t):
            tt = np.arange(nb) / nb
        else:
            tt = t

        th = np.mod(self.theta(tt), 2 * np.pi)
        tmp = np.diff(np.sign(s - th.reshape(-1, 1)), axis=0)
        chg, colk = np.where(tmp == -2)
        left = np.zeros(t.shape)
        left[colk] = tt[chg]
        right = np.ones(t.shape)
        right[colk] = tt[chg+1]

        done = np.abs(f(t, s)) < btol
        biter = 0

        self._log('Starting bisection ...')
        while not np.all(done) and biter < bmaxiter:
            biter = biter + 1
            t[~done] = 0.5 * (left[~done] + right[~done])
            fk = f(t[~done], s[~done])
            isneg = fk < 0
            left[~done] = isneg * left[~done] + ~isneg * t[~done]
            right[~done] = isneg * t[~done] + ~isneg * right[~done]
            done[~done] = np.abs(fk) < btol
        self._log('Bisection finished in %d steps' % biter)
        nmaxiter = 20

        fval = f(t, s)
        done = np.abs(fval) < ntol
        update = (~done).astype(np.float)
        prev_update = np.nan * np.ones(update.shape)
        niter = 0
        self._log('Starting Newton iteration ...\n')
        while not np.all(done) and niter < nmaxiter:
            niter = niter + 1

            update[~done] = fval[~done] / self.thetap(t[~done])
            t[~done] = t[~done] + update[~done]
            tmp1 = np.abs(prev_update[~done]) - np.abs(update[~done])
            if np.all(np.abs(tmp1) <= 100*eps()):
                break
            prev_update = update.copy()

            fval[~done] = f(t[~done], s[~done])
            done[~done] = np.abs(fval[~done]) < ntol
            update[done] = 0
        self._log('Newton iteration finished in %d steps...\n' % niter)
        self._log('label: %d/%d points with |f| > eps, max|f| = %.4f \n\n' % (np.sum(~done), np.size(t), np.max(np.abs(fval))))

        return t

    def thetap(self, ts):
        ts = np.asarray(ts).reshape(1, -1)[0, :]
        thp = 2 * np.pi / self.Saa * np.abs(self.phi(ts)**2)
        return thp

    def __str__(self):
        return 'Szego kernel object:\n\n'
