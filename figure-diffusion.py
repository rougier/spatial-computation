# Copyright (2017) Nicolas P. Rougier - BSD license
import voronoi

import numpy as np
from lxml import etree
import networkx as nx
import scipy.spatial.distance
import matplotlib.pyplot as plt

import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable



def connect(P, method="k-nearest", k=5, radius=1.0):
    n = len(P)
    dP = P.reshape(1,n,2) - P.reshape(n,1,2)
        
    # Distances
    D = np.hypot(dP[...,0], dP[...,1])
    I = np.argsort(D, axis=1)

    # Isotropic connections
    W = np.zeros((n,n))
    for i in range(n):
        # Connections (no self-connection -> 1:k+1)
        W[i,I[i,1:k+1]] = 1

    """
    # Angles
    A = np.zeros((n,n))
    for i in range(n):
        A[i] = np.arctan2(dP[i,:,1], dP[i,:,0]) * 180.0 / np.pi

    n = len(P)
    W = np.zeros((n,n))
    for i in range(n):
        p = 0
        for j in range(1,n):
            if -135 < A[i,I[i,j]] < -45:
                W[i,I[i,j]] = 1
                p += 1
            if p > k:
                break
    """
    return W

# ---------------------------------------------------------- Initialization ---
# SVG file where everything is defined
svg_filename = "data/gradient-1024x256.svg"
svg_tree = etree.parse(svg_filename).getroot()

# Points coordinates from the stippler program
dat_filename = "data/gradient-1024x256-stipple-1000.npy"
points = np.load(dat_filename)
X, Y = points[:,0], points[:,1]
n = len(points)

# Read dimensions from svg file directly
xmin, xmax = 0, int(svg_tree.get("width"))
ymin, ymax = 0, int(svg_tree.get("height"))

np.random.seed(123)

# ------------------------------------------------------------- Computation ---
# Connect neurons
W = connect(points, "k-nearest", k=5)

# Input sites
S1 = [[x, ymax] for x in np.linspace(xmin, xmax, 25, endpoint=True)]
S2 = [[xmin, y] for y in np.linspace(ymin, ymax, 10, endpoint=True)]


# Neurons (indices) receiving input
D = scipy.spatial.distance.cdist(S1, points)
I1 = np.argsort(D, axis=1)
I1 = np.unique(I1[:,0])

D = scipy.spatial.distance.cdist(S2, points)
I2 = np.argsort(D, axis=1)
I2 = np.unique(I2[:,0])

# Propagate increasing value from input sites
A1 = np.zeros(n)
for i in range(75):
    A1[I1] += 1
    A1 = np.maximum(A1, (A1*W).max(axis=1))
# Normalize "activity"
A1 = (A1 - A1.min()) / (A1.max() - A1.min())

A2 = np.zeros(n)
for i in range(150):
    A2[I2] += 1
    A2 = np.maximum(A2, (A2*W).max(axis=1))
# Normalize "activity"
A2 = (A2 - A2.min()) / (A2.max() - A2.min())


# ----------------------------------------------------------- Visualization ---
def plot_voronoi_activity(fig, ax, A):
    # Make space for colorbar inside ax
    divider = make_axes_locatable(ax)
    ax_colorbar = divider.new_horizontal(size="2%", pad=0.05)
    fig.add_axes(ax_colorbar)

    # Convert activity into colors
    facecolors = np.ones((n,4))
    cmap = plt.get_cmap('viridis') 
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap._A = []
    facecolors[:] = cmap.to_rgba(A)

    # Display voronoi cells
    patches = []
    regions, vertices = voronoi.voronoi_finite_polygons_2d(points)
    for region in regions:
        patches.append(Polygon(vertices[region]))
    collection = PatchCollection(patches, facecolors = facecolors,
                                 edgecolors = "black", linewidth = 0.25)
    ax.add_collection(collection)

    # Display colorbar
    colorbar = plt.colorbar(cmap, ticks=[0, 1], cax=ax_colorbar)
    colorbar.ax.set_yticklabels(['Past', 'Now'])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])



fig = plt.figure(figsize=(9.5,10))

# ------------------------------------------------------------------ Fig. 1 ---
ax = plt.subplot(4,1,1, aspect=1)
# These lines make this subplot the same size as subplot 3 & 4
divider = make_axes_locatable(ax)
ax_colorbar = divider.new_horizontal(size="2%", pad=0.05, frameon=False)
fig.add_axes(ax_colorbar)
ax_colorbar.set_yticks([])
ax_colorbar.set_xticks([])


# Show connections for some random units
n_samples = 12
samples = np.random.randint(0, n, n_samples)
s0, s1 = 25, 40
linewidth = 0.25
facecolors = np.ones((n,3))
edgecolors = np.zeros((n,3))
sizes = np.ones(n)*s0
linewidths = np.ones(n)*linewidth
for i in samples:
    facecolors[i] = 0,0,0
    edgecolors[i] = 1,1,1
    sizes[i] = s1
    linewidths[i] = 1
    for j in W[i].nonzero()[0]:
        facecolors[j] = 0,0,0
        edgecolors[j] = 1,1,1
        sizes[j] = s1
        linewidths[j] = 1

segments = []
for i in samples:
    src = points[i]
    for j in W[i].nonzero()[0]:
        tgt = points[j]
        segments.append([src, tgt])
c = LineCollection(segments, color="black", linewidth=1.0, zorder=-10)
ax.add_collection(c)

ax.scatter(X, Y, facecolor=facecolors, edgecolor=edgecolors,
           s=sizes, linewidth=linewidths)

ax.set_xlim(xmin, xmax)
ax.set_xticks([])
ax.set_ylim(ymin, ymax)
ax.set_yticks([])
ax.text(16, ymax-16, "A", ha="left", va="top", fontsize="24", weight="bold")


# ------------------------------------------------------------------ Fig. 2 ---
ax = plt.subplot(4,1,2, aspect=1)

# These 3 line make this subplot the same size as subplot 3 & 4
divider = make_axes_locatable(ax)
ax_colorbar = divider.new_horizontal(size="2%", pad=0.05, frameon=False)
fig.add_axes(ax_colorbar)
ax_colorbar.set_yticks([])
ax_colorbar.set_xticks([])

s0, s1 = 25, 40
linewidth = 0.25
facecolors = np.ones((n,3))
edgecolors = np.zeros((n,3))
sizes = np.ones(n)*s0
linewidths = np.ones(n)*linewidth
segments = []

ticks = []
labels = []

for x in np.linspace(xmin, xmax, 11, endpoint=False)[1:]:
    p = np.array([[x, ymax], [x, ymin]])
    d = scipy.spatial.distance.cdist(p,points)
    I = np.argsort(d, axis=1)
    # G = nx.from_numpy_matrix(W) #nx.Graph(W)
    G = nx.from_numpy_matrix(W * scipy.spatial.distance.cdist(points,points))
    i0, i1 = I[0,0], I[1,0]
    path = nx.shortest_path(G, i0, i1, 'weight')

    ticks.append(points[path[-1]][0])
    labels.append("n=%d" % (len(path)-1))
    
    for i in path:
        edgecolors[i] = 1,1,1
        facecolors[i] = 0,0,0
        sizes[i] = s1
        linewidths[i] = 1
        for i,j in zip(path[:-1],path[1:]):
            segments.append([points[i], points[j]])

c = LineCollection(segments, color="black", linewidth=1.0, zorder=-10)
ax.add_collection(c)

ax.scatter(X, Y, facecolor=facecolors, edgecolor=edgecolors,
           s=sizes, linewidth=linewidths)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_yticks([])

ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=8)
ax.text(16, ymax-16, "B", ha="left", va="top", fontsize="24", weight="bold")

# ------------------------------------------------------------------ Fig. 3 ---
ax = plt.subplot(4,1,3, aspect=1)
plot_voronoi_activity(fig, ax, A1)
for (x,y) in S1[1:-1]:
    ax.text(x, ymax, "↓", va="bottom", ha="center")
ax.text(16, ymax-16, "C", ha="left", va="top", fontsize="24", weight="bold")


# ------------------------------------------------------------------ Fig. 4 ---
ax = plt.subplot(4,1,4, aspect=1)
plot_voronoi_activity(fig, ax, A2)
for (x,y) in S2[1:-1]:
    ax.text(xmin, y, "→ ", va="center", ha="right")
ax.text(16, ymax-16, "D", ha="left", va="top", fontsize="24", weight="bold")

plt.tight_layout()

# plt.savefig("figures/figure-diffusion.pdf")
plt.show()
