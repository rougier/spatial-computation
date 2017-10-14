from .disk import Disk
from .circle import Circle


def unitdisk():
    """creates a unit disk region.

    d = unitdisk()
       Creates the unit disk region by d = disk(0, 1).
    """
    return Disk(Circle(0, 1))
