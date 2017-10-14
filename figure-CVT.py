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
import tqdm
import numpy as np
import scipy.spatial.distance
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
from matplotlib.collections import PatchCollection

import voronoi
from stippler import normalize, initialization


# Parameters
# ----------
n = 256
force = 0
xmin, xmax = 0, 1024
ymin, ymax = 0, 1024
density = np.ones((xmax-xmin,xmax-xmin))
density_P = density.cumsum(axis=1)
density_Q = density_P.cumsum(axis=1)

# Compute CVT
# -----------
if force:
    points = np.zeros((n,2))
    points[:,0] = np.random.uniform(xmin, xmax, n)
    points[:,1] = np.random.uniform(ymin, ymax, n)

    np.save("data/CVT-initial", points)
    for i in tqdm.trange(75):
        regions, points = voronoi.centroids(points, density, density_P, density_Q)
    np.save("data/CVT-final.npy", points)


# Display
# -------
plt.figure(figsize=(10,5))

# ------------------------------------------------------------------ Fig 1. ---
ax = plt.subplot(1, 2, 1, aspect=1)
points = np.load("data/CVT-initial.npy")

patches = []
regions, vertices = voronoi.voronoi_finite_polygons_2d(points)
for region in regions:
    patches.append(Polygon(vertices[region]))
collection = PatchCollection(patches,
                             facecolor="white", edgecolor="black", linewidth=0.25)
ax.add_collection(collection)

ax.scatter(points[:,0], points[:,1], s=5,
           facecolor="black", edgecolor="none")
regions, points = voronoi.centroids(points, density, density_P, density_Q)
ax.scatter(points[:,0], points[:,1], s=20,
           facecolor="none", edgecolor="black", linewidth=.5)

ax.text(24, ymax-24, "A", color="black", weight="bold", va="top", fontsize=24)

ax.set_xlim(xmin, xmax)
ax.set_xticks([])
ax.set_ylim(ymin, ymax)
ax.set_yticks([])


# ------------------------------------------------------------------ Fig 2. ---
ax = plt.subplot(1, 2, 2, aspect=1)
points = np.load("data/CVT-final.npy")

patches = []
regions, vertices = voronoi.voronoi_finite_polygons_2d(points)
for i,region in enumerate(regions):
    patches.append(Polygon(vertices[region]))
collection = PatchCollection(patches,
                             facecolor="white", edgecolor="black", linewidth=0.25)
ax.add_collection(collection)

ax.scatter(points[:,0], points[:,1], s=5,
           facecolor="black", edgecolor="none")
regions, points = voronoi.centroids(points, density, density_P, density_Q)
ax.scatter(points[:,0], points[:,1], s=20,
           facecolor="none", edgecolor="black", linewidth=.5)

ax.text(24, ymax-24, "B", color="black", weight="bold", va="top", fontsize=24)

ax.set_xlim(xmin, xmax)
ax.set_xticks([])
ax.set_ylim(ymin, ymax)
ax.set_yticks([])


plt.tight_layout()
# plt.savefig("figures/figure-CVT.pdf")
plt.show()
