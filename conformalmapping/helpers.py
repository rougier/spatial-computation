import functools
import numpy as np


def suppress_warnings(f):
    """Function decorator to prevent numpy raising warnings
    """
    @functools.wraps(f)
    def impl(*args, **kwargs):
        oldsettings = {}
        try:
            oldsettings = np.seterr(all='ignore')
            return f(*args, **kwargs)
        finally:
            np.seterr(**oldsettings)
    return impl


def eps(z=1):
    """Wrapper around spacing that works for complex numbers
    """
    zre = np.abs(np.real(z))
    zim = np.abs(np.imag(z))
    return np.spacing(np.max([zre, zim]))


def flipud(a):
    return a[::-1]
