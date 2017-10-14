import numpy as np

def boundbox(points):
    """Calculates the bounding box around complex points.
    
    box = boundbox(points) calculates a bounding box around a set of points
    in the complex plane, and returns coordinates in AXIS format.
    
    See also axis, plotbox.
    """
    points = np.asarray(points, dtype = np.complex)
    minx = np.min(points.real)
    miny = np.min(points.imag)
    maxx = np.max(points.real)
    maxy = np.max(points.imag)
    return np.array([minx, maxx, miny, maxy], dtype = np.double)

def plotbox(points, scale = 1.2):
    """Expand a box by a scaling factor
    """
    box = boundbox(points)

    dx = (box[1] - box[0])
    dy = (box[3] - box[2])
    dbox = (scale/2.0) * np.max([dx, dx])
    dbox = dbox * np.array([-1.0, 1.0])

    midx = np.mean([box[0], box[1]])
    midy = np.mean([box[2], box[3]])

    return np.hstack([midx + dbox, midy + dbox])

def bb2z(box):
    """Axis bounding box to vertices
    """
    z = [ np.complex(box[i], box[j]) for i, j in zip([0,1,1,0], [2,2,3,3]) ]
    z = np.array(z)
    return z
