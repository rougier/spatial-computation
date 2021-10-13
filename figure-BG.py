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
import imageio
import geometry
import numpy as np
from lxml import etree
import scipy.ndimage
from voronoi import voronoi_finite_polygons_2d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



# ---------------------------------------------------------- Initialization ---
# SVG file where everything is defined
svg_filename = "data/BG-1024x1024.svg"
svg_tree = etree.parse(svg_filename).getroot()

# Bitmap export of the density layer in the SVG file
img_filename = "data/BG-1024x1024-density.png"
img = imageio.imread(img_filename).astype(int)
img_density = img[...,3]
img_identity = img[...,0]*256*256 + img[...,1]*256 + img[...,2]

# Points coordinates from the stippler program
dat_filename = "data/BG-1024x1024-stipple-2500.npy"
# points = np.load(dat_filename)
global_coords = np.load(dat_filename)
n = len(global_coords)
local_coords = np.zeros((n,2))
facecolors = np.zeros((n,4))
facecolors[...] = (1,1,1,1)
edgecolors = np.zeros((n,4))
edgecolors[...] = (0,0,0,1)
activity = np.zeros(n)


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
                print(name, end-start)
                group["coords"]["global"] = global_coords[start:end]
                group["facecolors"]       = facecolors[start:end]
                group["edgecolors"]       = edgecolors[start:end]
                group["activity"]         = activity[start:end]
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
    # m = max(np.abs(X).max(), np.abs(Y).max())
    # X /= m
    # Y /= m
    
    groups[name]["coords"]["local"] = np.dstack([X,Y]).squeeze()
    return groups[name]["coords"]["local"]

compute_local_coordinates(groups, "Caudate")
compute_local_coordinates(groups, "GPi")
compute_local_coordinates(groups, "GPe")


def compute_fake_activation(groups, name, center=(0,0), sigma=5):
    """ """
    P = groups[name]["coords"]["local"] - center
    X, Y = P[:,0], P[:,1]
    D = np.sqrt(X*X+Y*Y)
    D /= D.max()
    D = np.exp(-sigma*D*D)
    groups[name]["activity"][...] = D
    
compute_fake_activation(groups, "Caudate", center=(-0.00,0.5))
compute_fake_activation(groups, "GPe", center=(-0.5,+0.25))
compute_fake_activation(groups, "GPi", center=(-0.25,-0.5))



# ----------------------------------------------------------- Visualization ---
def plot_histogram(ax, points, W=None, bins=(32,32), **kwargs):
    """ """
    X, Y = points[:,0], points[:,1]
    kwargs["interpolation"] = kwargs.get("interpolation") or "bicubic"
    kwargs["origin"] = kwargs.get("origin", "lower")
    kwargs["extent"] = kwargs.get("extent", [xmin, xmax, ymin, ymax])
    kwargs["cmap"] = kwargs.get("cmap", plt.get_cmap('viridis'))
    H, _, _ = np.histogram2d(X, Y, bins=bins, weights=W, 
                             range=[[xmin, xmax], [ymin, ymax]])
                             
    return ax.imshow(H.T, **kwargs)

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
    default_size = 15
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

    
def plot_voronoi(ax, points, paths, **kwargs):
    """ """

    patches = []
    regions, vertices = voronoi_finite_polygons_2d(points)
    for region in regions:
        patches.append(Polygon(vertices[region]))

    options = { "facecolor": 'none', "edgecolor": 'none' }
    kwargs["zorder"] = kwargs.get("zorder", -10)
    kwargs["facecolors"] = kwargs.get("facecolor", facecolors)
    kwargs["edgecolors"] = kwargs.get("edgecolor", edgecolors)
    kwargs["linewidth"] = kwargs.get("linewidth", 0.25)

    for path in paths:
        patch = PathPatch(path, **options)
        ax.add_patch(patch)
        collection = PatchCollection(patches, **kwargs)
        collection.set_clip_path(patch)
        ax.add_collection(collection)




plt.figure(figsize=(10,10))

# ------------------------------------------------------------------ Fig. 1 ---
ax = plt.subplot(2,2,1, aspect=1, facecolor="white")
# img_filename = "data/BG-1024x1024-density.png"
img_filename = "data/BG-1024x1024-bis-density.png"
img = imageio.imread(img_filename)
ax.imshow(img, origin='lower', extent=[xmin,xmax,ymin,ymax], interpolation="nearest")

paths = [groups["Caudate"]["border"],
         groups["GPe"]["border"],
         groups["GPi"]["border"]]
plot_paths(ax, paths, edgecolor="black", zorder=10)

paths = [groups["Caudate"]["major-axis"],
         groups["GPe"]["major-axis"],
         groups["GPi"]["major-axis"]]
plot_paths(ax, paths, edgecolor="black", linestyle="--")

paths = [groups["Caudate"]["minor-axis"],
         groups["GPe"]["minor-axis"],
         groups["GPi"]["minor-axis"]]
plot_paths(ax, paths, edgecolor="black", linestyle="--")

paths = [groups["Caudate"]["input"],]
plot_paths(ax, paths, edgecolor="red", linewidth=2.0)

paths = [groups["GPi"]["output"],]
plot_paths(ax, paths, edgecolor="blue", linewidth=2.0)

ax.set_xlim(xmin,xmax)
ax.set_xticks([])
ax.set_ylim(ymax, ymin)
ax.set_yticks([])
ax.text(32, 32, "A", ha="left", va="top", fontsize="24", weight="bold")


# ------------------------------------------------------------------ Fig. 2 ---
paths = [groups["Caudate"]["border"],
         groups["GPe"]["border"],
         groups["GPi"]["border"]]

ax = plt.subplot(2,2,2, aspect=1, facecolor="white")
# plot_paths(ax, paths, edgecolor="black")

facecolors[...] = (1,1,1,1)
edgecolors[...] = (0,0,0,1)

path = svg.path(svg_filename, "Caudate-input")
verts, codes = svg.tesselate(path.vertices, path.codes)
D = np.abs(geometry.signed_distance_polyline(verts, global_coords))
I = D.argsort()
red = (0.83,0.15,0.15,1.00)
facecolors[I[:200]] = red
edgecolors[I[:200]] = red

path = svg.path(svg_filename, "GPi-output")
verts, codes = svg.tesselate(path.vertices, path.codes)
D = np.abs(geometry.signed_distance_polyline(verts, global_coords))
I = D.argsort()
blue = (0.12,0.46,0.70,1.00)
facecolors[I[:40]] = blue
edgecolors[I[:40]] = blue

red = ax.scatter([], [], color=red, s=5, label='Input')
blue = ax.scatter([], [], color=blue, s=5, label='Output')
handles = [red, blue]
labels = [h.get_label() for h in handles] 
ax.legend(handles=handles, labels=labels, scatterpoints=3, frameon=False)

plot_points(ax, global_coords, facecolor=facecolors, edgecolor=edgecolors)



# plot_points(ax, global_coords, facecolor="white", edgecolor="black")
ax.set_xlim(xmin,xmax)
ax.set_xticks([])
ax.set_ylim(ymax, ymin)
ax.set_yticks([])
ax.text(32, 32, "B", ha="left", va="top", fontsize="24", weight="bold")

facecolors[...] = (1,1,1,1)
edgecolors[...] = (0,0,0,1)


# ------------------------------------------------------------------ Fig. 3 ---
ax = plt.subplot(2,2,3, aspect=1, facecolor="white")
plot_paths(ax, paths)
im = plot_histogram(ax, global_coords, W=activity, bins=(32,32))
ax.set_xlim(xmin,xmax)
ax.set_xticks([])
ax.set_ylim(ymax, ymin)
ax.set_yticks([])
ax.text(32, 32, "C", ha="left", va="top", fontsize="24", weight="bold", color="white")

axins = inset_axes(ax, width="3%",  height="33%", loc=1)
colorbar = plt.colorbar(im, cax=axins, orientation="vertical", ticks=[])
colorbar.outline.set_edgecolor('white')
colorbar.outline.set_linewidth(0.5)
colorbar.set_label("Relative activity", color="white")
colorbar.ax.yaxis.set_label_position('left')

# ------------------------------------------------------------------ Fig. 4 ---
ax = plt.subplot(2,2,4, aspect=1, facecolor="white")
ax.set_facecolor("black")
cmap = plt.get_cmap('viridis')
norm = matplotlib.colors.Normalize(vmin=activity.min(), vmax=activity.max())
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
cmap._A = []
facecolors[...] = cmap.to_rgba(activity)
plot_voronoi(ax, global_coords, paths)
plot_paths(ax, paths, edgecolor="white", linewidth=1.0, zorder=10)

axins = inset_axes(ax, width="3%",  height="33%", loc=1)
colorbar = plt.colorbar(cmap, cax=axins, orientation="vertical", ticks=[])
colorbar.outline.set_edgecolor('white')
colorbar.outline.set_linewidth(0.5)
colorbar.set_label("Relative activity", color="white")
colorbar.ax.yaxis.set_label_position('left')

ax.set_xlim(xmin,xmax)
ax.set_xticks([])
ax.set_ylim(ymax, ymin)
ax.set_yticks([])
ax.text(32, 32, "D", ha="left", va="top", fontsize="24", weight="bold", color="white")

plt.tight_layout()
# plt.savefig("figures/figure-BG.pdf")
plt.show()
