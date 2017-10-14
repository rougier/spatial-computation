import matplotlib.pyplot as plt
from ._compat import *


class GridCurves(object):

    def __init__(self, curves):
        self._curves = curves

    @property
    def curves(self):
        return self._curves

    def __str__(self):
        fh = StringIO()
        fh.write('gridcurves objects:\n\n')
        fh.write('  with %d gridlines.:\n\n' % len(self.curves))
        return fh.getvalue()

    def __repr__(self):
        return str(self)

    def plot(self, *args, **kwargs):
        for curve in self.curves:
            plt.plot(curve.real, curve.imag, color = '0.6',lw=.5)
