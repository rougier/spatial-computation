import numpy as np
from .region import Region
from .gridcurves import GridCurves

class Disk(Region):
    def __init__(self, circle, *args, **kwargs):
        super(Disk, self).__init__(outer = circle, **kwargs)
        self._gridType = 'polar'
        self._numRadialLines = 2*20
        self._numCircularLines = 2*5
        self._numLevels = 5

    @property
    def gridType(self):
        return self._gridType

    @gridType.setter
    def gridType(self, value):
        assert(value in ['polar', 'carleson'])
        self._gridType = value

    @property
    def numRadialLines(self):
        return self._numRadialLines

    @numRadialLines.setter
    def numRadialLines(self, value):
        assert(type(value) == int and value > 0)
        self._numRadialLines = value

    @property
    def numCircularLines(self):
        return self._numCircularLines

    @numCircularLines.setter
    def numCircularLines(self, value):
        assert(type(value) == int and value > 0)
        self._numCircularLines = value

    @property
    def numLevels(self):
        return self._numLevels

    @numLevels.setter
    def numLevels(self, value):
        assert(type(value) == int and value > 0)
        self._numLevels = value

    def grid(self, **kwargs):
        if self.gridType == 'polar':
            return self.polarGrid(**kwargs)
        else:
            raise NotImplementedError('Unknown grid type')

    def polarGrid(self):

        nrad  = self.numRadialLines
        ncirc = self.numCircularLines

        nrad  = 36
        ncirc = 16

        npt = 512
        c = self.outer.center
        r = self.outer.radius
        curves = []

        if 1:
            for a in np.linspace(-r,r,2*16):
                b = np.sqrt(r*r-a*a)
                curves.append(c + a*np.ones(npt) + 1j*np.linspace(-b,b,npt))
                curves.append(c + np.linspace(-b,b,npt) + 1j*a*np.ones(npt))
        else:
            zg = np.arange(1, npt+1) / float(npt+1)
            for k in range(nrad):
                crv = c + r * np.exp(2j * np.pi * k/float(nrad)) * zg
                curves.append(crv)

            zg = np.exp(2j * np.pi * np.arange(npt)/float(npt-1))
            for k in range(1, ncirc+1):
                crv = c + r*k/float(ncirc+1)*zg
                curves.append(crv)

        return GridCurves(curves)
