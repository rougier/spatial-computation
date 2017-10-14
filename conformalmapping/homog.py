import numpy as np


class Homog(object):
    # Note that in the original Driscoll and Kropf refer to
    # homog being compatible with `double`. In a Python sense,
    # this is actually a complex. In MATLAB, a complex number is-a
    # double

    # Also, the numpy representation of vectors and matrices is
    # just different enough that homog is never quite going to be
    # first class.

    def __init__(self, z1, z2=np.complex(1.0, 0.0)):
        self._numerator = z1
        self._denominator = z2

    def isinf(self):
        # TODO - confirm that this test is correct in python
        zero = 0.0 + 0.0j
        return (self._denominator == zero) and (self._numerator != zero)

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    def __complex__(self):
        if self.isinf():
            # follows the convention in the original MATLAB of returning real infinity
            return np.complex(np.inf)
        else:
            return self._numerator / self._denominator

    def __abs__(self):
        return self.__complex__().__abs__()

    def __str__(self):
        return '%s / %s' % (self._numerator, self._denominator)

    # TODO __add__
    # TODO __radd__
    # TODO other operators ...
