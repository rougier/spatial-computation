import numpy as np
from .closedcurve import ClosedCurve
from . import cmt


class Region(object):
    """A region in the complex plane

    A region is defined by one or more closed curves. The interior of a
    region is defined to be to the left of the tangent vector of the closed
    curve. The exterior is the compliment of the interior.
    """

    def __init__(self, outer=None, inner=None, *args, **kwargs):
        self._outerboundaries = self._checkcc(outer)
        self._innerboundaries = self._checkcc(inner)

    @property
    def outer(self):
        if len(self._outerboundaries) == 1:
            return self._outerboundaries[0]
        else:
            return self._outerboundaries

    @property
    def inner(self):
        if len(self._innerboundaries) == 1:
            return self._innerboundaries[0]
        else:
            return self._innerboundaries

    @property
    def numinner(self):
        return len(self._innerboundaries)

    @property
    def numouter(self):
        return len(self._outerboundaries)

    @property
    def boundaries(self):
        """Return a combined list of (inner and outer) boundaries
        """
        return self._outerboundaries + self._innerboundaries

    @property
    def connectivity(self):
        return self.numinner + self.numouter

    @staticmethod
    def interiorto(*args):
        # https://github.com/tobydriscoll/conformalmapping/blob/master/region.m#L68
        pass

    @staticmethod
    def exteriorto(*args):
        # https://github.com/tobydriscoll/conformalmapping/blob/master/region.m#L70
        pass

    def _checkcc(self, suitor):
        """Validate a given boundary

        Parameters
        ----------
        suitor : None, [ClosedCurve] or ClosedCurve
            inputs

        Returns
        -------
        out : [ClosedCurve]
            the actual boundaries in the internal representation
        """
        if suitor is None:
            return []
        elif isinstance(suitor, ClosedCurve):
            return [suitor]
        else:
            for item in suitor:
                if not isinstance(item, ClosedCurve):
                    raise Exception('Not a list of closed curves')
            return suitor
        raise Exception('Not a list of closed curves')

    def isempty(self):
        return (self._innerboundaries != []) and (self._outerboundaries != [])

    def hasinner(self):
        return len(self._innerboundaries) > 0

    def hasouter(self):
        return len(self._outerboundaries) > 0

    def isexterior(self):
        return self.hasinner() and not self.hasouter()

    def isinterior(self):
        return self.hasouter() and not self.hasinner()

    def issimplyconnected(self):
        if self.isexterior() and self.numinner == 1:
            return True
        if self.isinterior() and self.numouter == 1:
            return True
        return False

    def boundbox(self):
        # The bounding box of a region is the bounding-box which encloses
        # all of the boundaries (inner and outer) that make up the region.
        zi = []
        for b in self.boundaries:
            zi.append(cmt.bb2z(b.boundbox()))
        zi = np.vstack(zi)
        zi = np.max(zi, axis=0)
        return zi

    def __str__(self):
        if self.isempty():
            return 'empty region'
        else:
            return 'non-empty Region'

    def grid(self):
        return self.grid

    def plot(self, **kwargs):
        # This is reasonably simple minded until adapt is complete

        for b in self._innerboundaries:
            b.plot(**kwargs)

        for b in self._outerboundaries:
            b.plot(**kwargs)
