# -----------------------------------------------------------------------------
# Copyright (c) 2017, Nicolas P. Rougier. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
import svg
import voronoi
import geometry
import numpy as np
from lxml import etree
import scipy.ndimage
from voronoi import voronoi_finite_polygons_2d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# from matplotlib.patches import PathPatch, Polygon
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes



# ---------------------------------------------------------- Initialization ---
# SVG file where everything is defined
svg_filename = "data/BG-1024x1024.svg"
svg_tree = etree.parse(svg_filename).getroot()

# Bitmap export of the density layer in the SVG file
img_filename = "data/BG-1024x1024-density.png"
img = scipy.misc.imread(img_filename, mode='RGBA').astype(int)
img_density = img[...,3]
img_identity = img[...,0]*256*256 + img[...,1]*256 + img[...,2]

# Points coordinates from the stippler program
dat_filename = "data/BG-1024x1024-stipple-2500.npy"
global_coords = np.load(dat_filename)
n = len(global_coords)
local_coords = np.zeros((n,2))
facecolors = np.zeros((n,4))
facecolors[...] = (1,1,1,1)
edgecolors = np.zeros((n,4))
edgecolors[...] = (0,0,0,1)
sizes = np.zeros(n)
sizes[...] = 40


# Read dimensions from svg file directly
xmin, xmax = 0, int(svg_tree.get("width"))
ymin, ymax = 0, int(svg_tree.get("height"))

# Group identity (color) defined within the SVG file
# (note that we could also extract this from the SVG file)
base = np.array([256*256, 256, 1], dtype=int)
groups = { "Caudate" : { "id" :  ((255, 0, 0)*base).sum(),
                         "coords" : { } },
           "GPi"     : { "id" :  ((0, 255, 0)*base).sum(),
                         "coords" : { } },
           "GPe"     : { "id" :  ((0, 0, 255)*base).sum(),
                         "coords" : { } }
         }

# We sort the points array by color such that group points are contiguous
X, Y = global_coords[:,0], global_coords[:,1]
ids = img_identity[Y.astype(int),X.astype(int)]
I = np.argsort(ids)
start, end = 0,0
for i in range(1, len(ids)):
    if ids[I[i]] != ids[I[i-1]] or i == len(ids)-1:
        if i == len(ids)-1:
            end = i+1
        else:
            end = i
        for name,group in groups.items():
            if group["id"] == ids[I[start]]:
                group["indices"] = start,end
                group["coords"]["global"] = global_coords[start:end]
                group["facecolors"]       = facecolors[start:end]
                group["edgecolors"]       = edgecolors[start:end]
        start = i
global_coords[...] = global_coords[I]


# Get individual paths
groups["Caudate"]["border"] = svg.path(svg_filename, "Caudate")
groups["Caudate"]["major-axis"] = svg.path(svg_filename, "Caudate-major-axis")
groups["Caudate"]["minor-axis"] = svg.path(svg_filename, "Caudate-minor-axis")
groups["Caudate"]["input"] = svg.path(svg_filename, "Caudate-input")
groups["GPe"]["border"] = svg.path(svg_filename, "GPe")
groups["GPe"]["major-axis"] = svg.path(svg_filename, "GPe-major-axis")
groups["GPe"]["minor-axis"] = svg.path(svg_filename, "GPe-minor-axis")
groups["GPi"]["border"] = svg.path(svg_filename, "GPi")
groups["GPi"]["major-axis"] = svg.path(svg_filename, "GPi-major-axis")
groups["GPi"]["minor-axis"] = svg.path(svg_filename, "GPi-minor-axis")
groups["GPi"]["output"] = svg.path(svg_filename, "GPi-output")


# ------------------------------------------------------------- Computation ---
def compute_local_coordinates(groups, name):
    """
    Compute local coordinates of a structure relatively to major and minor axes.
    """

    points = groups[name]["coords"]["global"]

    # Distance to major axis (X)
    path = svg.path(svg_filename, "%s-major-axis" % name)
    verts, codes = svg.tesselate(path.vertices, path.codes)
    X = geometry.signed_distance_polyline(verts, points)

    # Distance to minor axis (Y)
    path = svg.path(svg_filename, "%s-minor-axis" % name)
    verts, codes = svg.tesselate(path.vertices, path.codes)
    Y = geometry.signed_distance_polyline(verts, points)

    # Differetial normalization
    X /= np.abs(X).max()
    Y /= np.abs(Y).max()

    # Common normalization
    #m = max(np.abs(X).max(), np.abs(Y).max())
    #X /= m
    #Y /= m
    
    groups[name]["coords"]["local"] = np.dstack([X,Y]).squeeze()
    return groups[name]["coords"]["local"]

compute_local_coordinates(groups, "Caudate")
compute_local_coordinates(groups, "GPi")
compute_local_coordinates(groups, "GPe")



# ----------------------------------------------------------- Visualization ---

def plot_paths(ax, paths, **kwargs):
    """ """
    kwargs["facecolor"] = kwargs.get("facecolor", "none")
    kwargs["edgecolor"] = kwargs.get("edgecolor", "white")
    kwargs["linewidth"] = kwargs.get("linewidth", 1.0)
    for path in paths:
        ax.add_patch(PathPatch(path, **kwargs))

def plot_points(ax, points, **kwargs):
    """ """
    reference = 1000
    default_size = 75
    default_linewidth = .75
    ratio = reference/len(points)
    size = max(ratio * default_size, 3.0)
    linewidth = max(ratio * default_linewidth, 0.25)
    X, Y = points[:,0], points[:,1]
    kwargs["facecolor"] = kwargs.get("facecolor", "none")
    kwargs["edgecolor"] = kwargs.get("edgecolor", "white")
    kwargs["linewidth"] = kwargs.get("linewidth", linewidth)
    kwargs["s"] = kwargs.get("s") or size
    ax.scatter(X, Y, **kwargs)

def plot_groups(c, color):
    
    start, end = groups["Caudate"]["indices"]
    n1 = end-start
    start, end = groups["GPe"]["indices"]
    n2 = end-start
    r1 = n2/n1
    start, end = groups["GPi"]["indices"]
    n3 = end-start
    r2 = n3/n1

    n1 = 50
    n2 = int(np.floor(n1*r1))
    n3 = int(np.floor(n1*r2))

    D = scipy.spatial.distance.cdist(groups["Caudate"]["coords"]["local"], [c])
    indices = np.argsort(D,axis=0)[:n1]
    groups["Caudate"]["facecolors"][indices] = color
    groups["Caudate"]["edgecolors"][indices] = color
    D = scipy.spatial.distance.cdist(groups["GPe"]["coords"]["local"], [c])
    indices = np.argsort(D,axis=0)[:n2]
    groups["GPe"]["facecolors"][indices] = color
    groups["GPe"]["edgecolors"][indices] = color
    D = scipy.spatial.distance.cdist(groups["GPi"]["coords"]["local"], [c])
    indices = np.argsort(D,axis=0)[:n3]
    groups["GPi"]["facecolors"][indices] = color
    groups["GPi"]["edgecolors"][indices] = color

    

plt.figure(figsize=(8,8))

ax = plt.subplot(1,1,1, aspect=1, facecolor="white")

plot_groups((+0.50, -0.75), (1,0,0,1))
plot_groups((-0.25, +0.00), (0,0,1,1))
plot_groups((-0.50, +0.75), (.5,.5,.5,1))
    
plot_points(ax, global_coords, facecolor=facecolors, edgecolor=edgecolors)

ax.set_xlim(xmin,xmax)
ax.set_xticks([])
ax.set_ylim(ymax, ymin)
ax.set_yticks([])

plt.tight_layout()
# plt.savefig("figures/BG-connection.pdf")
plt.show()
