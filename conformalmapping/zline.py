import numpy as np
from .curve import Curve


class Zline(Curve):

    def __init__(self, points, **kwargs):
        self._points = points
        self._tangent = np.diff(points)

        def position(t):
            ts = t / (1-t**2)
            return self._points[0] + ts * self._tangent[0]

        def tangent(t):
            return self._tangent

        super(Zline, self).__init__(positionfun=position, 
                                    tangentfun=tangent,
                                    bounds=(-1.0, 1.0))

    def __str__(self):
        out = 'line passing through %s and tangent to %s' % (self._points[0], self.tangent(0))
        return out
