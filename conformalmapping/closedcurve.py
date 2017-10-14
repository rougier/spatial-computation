import matplotlib.pyplot as plt
import numpy as np
from .curve import Curve


class ClosedCurve(Curve):
    """Base class for simple planar Jordan curves
    """

    def __init__(self, *args, **kwargs):
        super(ClosedCurve, self).__init__(*args, **kwargs)

    def modparam(self, ts):
        tmin, tmax = self.bounds
        dt = tmax - tmin
        ts = ts - tmin
        return tmin + np.mod(ts - tmin, dt)

    def __str__(self):
        return 'closedcurve'

    def plot(self):
        ts = np.linspace(0.0, 1.0, 300)
        zs = self.point(ts)
        plt.plot(zs.real, zs.imag, 'k-')
