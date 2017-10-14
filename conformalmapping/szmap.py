import numpy as np
from .conformalmap import ConformalMap
from .closedcurve import ClosedCurve
from .unitdisk import unitdisk
from .region import Region
from .szego import Szego, SzegoOpts


class SzMap(ConformalMap):
    """SzMap represents a Riemann map via the Szego kernel.
    """

    def __init__(self, range=None, conformalCenter=0, **kwargs):
        """Create a new conformal map based on the Szego kernel

        Parameters
        ----------
        range : Region or ClosedCurve
            an object that represents the range of the map

        conformalCenter : complex
            the conformal center (forward to the szego kernel)
        """
        if isinstance(range, ClosedCurve):
            range = Region(range)
        if not range.issimplyconnected():
            raise Exception('Region must be simply connected')
        kwargs['range'] = range
        kwargs['domain'] = unitdisk()

        super(SzMap, self).__init__(**kwargs)

        boundary = self.range.outer

        # question, how to alter these?
        szargs = SzegoOpts()
        S = Szego(boundary, conformalCenter, szargs)

        nF = szargs.numFourierPts
        t = S.invtheta(2*np.pi*np.arange(nF)/float(nF))
        c = np.fft.fft(boundary(t))/float(nF)
        c = c[::-1]
        self._kernel = S
        self._coefficients = c
        self._opts = szargs

    def applyMap(self, z):
        return np.polyval(self._coefficients, z)
