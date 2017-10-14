import numpy as np
from .closedcurve import ClosedCurve
from .mobiusbase import MobiusBase, standardmap
from .zline import Zline
from ._compat import *


class Circle(ClosedCurve):
    """Circle is a generalized circle class.

    Circle is generalized in that it can represent either a circle,
    or a circle with infinite radius (a line). This representation
    is necessary to make the math come out nicely.
    """

    def __init__(self, center=np.nan, radius=np.inf, line=None):
        """Creates a circle with given center and radius.

        Parameters
        ----------
        circle : complex
            the location of the circle in the complex plane

        radius : real
            the radius of the circle
        """
        # TODO - the type checking here is because of minor python 2/3
        #        compatibility issue. Remove these checks when that issue
        #        is resolved
        if type(center) == np.ndarray:
            center = center[0]
        if type(radius) == np.ndarray:
            radius = radius[0]
        if radius < 0.0:
            raise ValueError('Circle must have a postive radius')
        self._center = center
        self._radius = radius
        self._line = line

        def position(ts):
            return center + radius * np.exp(1.0j * ts)

        def tangent(ts):
            return 1.0j * np.exp(1.0j * ts)

        super(Circle, self).__init__(positionfun=position,
                                     tangentfun=tangent,
                                     bounds=(0.0, 2.0 * np.pi))

    @staticmethod
    def from_points(z1, z2, z3):
        """Creates a generalized circle passing through three given points
        """
        return Circle.from_vector([z1, z2, z3])

    @staticmethod
    def from_vector(z3):
        """Create a generalized circle passing through three points

        Parameters
        ----------
        z3 : array-like
            a three element array of complex
        """
        z3 = np.asarray(z3).astype(np.complex)
        # TODO - check that the array has exactly three elements
        M_a = standardmap(z3)
        M_b = standardmap([1.0, 1.0j, -1.0])
        M = MobiusBase(np.linalg.solve(M_a, M_b))
        zi = M.pole()
        if np.abs(np.abs(zi) - 1) < 10*np.spacing(1):
            if np.all(np.isreal(z3)):
                z3.sort()
            return Circle(line=Zline(z3[:2]))
        else:
            center = M(1.0/zi.conjugate())
            radius = np.abs(z3[0] - center)
            return Circle(center=center, radius=radius)

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    @property
    def line(self):
        return self._line

    def point(self, t):
        if not self.isinf():
            z = self._center + self._radius * np.exp(1.0j * t * 2*np.pi)
            return z
        else:
            raise NotImplementedError('Circle.point for the line case not implemented')

    def __str__(self):
        fh = StringIO()
        if self.isinf():
            fh.write('circle (generalized) as a line, \n')
        else:
            fh.write('circle with centre %s and radius %s,\n' % (self.center, self.radius))
        return fh.getvalue()

    def __repr__(self):
        return str(self)

    def dist(self, z):
        """Distance between this circle and a point
        """
        if not self.isinf():
            v = z - self.center
            d = np.abs(np.abs(v) - self.radius)
            return d
        else:
            v = z - self.line.position(0)
            return None

    def fill(self):
        pass

    def isinf(self):
        """True if the circle is really a line
        """
        return np.isinf(self.radius)

    def isinside(self, z):
        """True if the point is inside the circle

        In the case where the circle has infinite radius, isinside
        is still a valid operation, and will return true for a halfplane

        Parameters
        ----------
        z : complex
            the point to check if it is inside
        """
        if self.isinf():
            z0 = self.line.position(0)
            z = (z - z0) / self.line.tangent(0)
            return z.imag > 0
        else:
            return np.abs(z - self._center) < self._radius
