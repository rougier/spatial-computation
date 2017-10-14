# This file is a part of the CMToolbox.
# It is licensed under the BSD 3-clause license.
# (See LICENSE.)
#
# Copyright Toby Driscoll, 2014.
# (Re)written by Everett Kropf, 2014,
# adapted from code by Toby Driscoll, originally 20??.
# Python Port Copyright Andrew Walker, 2015
import numpy as np
from .conformalmap import ConformalMap
from .homog import Homog


class MobiusBase(ConformalMap):
    """Mobius transformation class
    """
    def __init__(self, M=None, **kwargs):
        """Takes a 2 x 2 matrix M of doubles
        """
        if M is None:
            self._M = None
        else:
            assert(M.shape == (2, 2))
            self._M = M
        super(MobiusBase, self).__init__(**kwargs)

    @property
    def matrix(self):
        return self._M

    def __str__(self):
        if self._M is None:
            return '\n\tempty transformation matrix\n\n'
        else:
            return '\n\tmobius transform (%s)\n\n' % str(self._M)

    def applyMap(self, z):
        if isinstance(z, np.ndarray):
            # TODO - vectorize
            return np.array([self(i) for i in z])
        elif np.isscalar(z):
            z0 = complex(z)
            z = Homog(z0)
            Z = np.array([[z.numerator, z.denominator]], dtype=np.complex).T
            W = np.dot(self._M, Z)
            w = Homog(W[0], W[1])
            w = w.__complex__()
            return w
        else:
            raise Exception('conformal map not known for %s' % type(z))

    def pole(self):
        A = self._M
        z = Homog(-A[1, 1], A[1, 0])
        return z.__complex__()

    def zero(self):
        A = self._M
        z = Homog(-A[0, 2], A[0, 0])
        return z.__complex__()


def standardmap(z):
    # make sure that it's an array
    z = np.asarray(z, dtype=np.complex)

    # make sure it's the right size
    z = z.reshape(3, 1)[:, 0]

    if np.isinf(z[0]):
        return np.array([[0, z[1]-z[2]], [1, -z[2]]])
    elif np.isinf(z[1]):
        return np.array([[1, -z[0]], [1, -z[2]]])
    elif np.isinf(z[2]):
        return np.array([[1, -z[0]], [0, z[1]-z[0]]])
    else:
        rms = z[1] - z[2]
        rmq = z[1] - z[0]
        return np.array([[rms, -z[0]*rms], [rmq, -z[2]*rmq]])
