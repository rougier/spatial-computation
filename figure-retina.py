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
from scipy import interpolate
import scipy.spatial.distance
import voronoi
from stippler import normalize, initialization
from voronoi import voronoi_finite_polygons_2d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path

def polygon_area(P):
    lines = np.hstack([P,np.roll(P,-1,axis=0)])
    return 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))


def blob(center, radius):
    n = 10
    noise = 0.4
    T = np.linspace(0, 2*np.pi, n, endpoint=False)
    R = np.random.uniform(1-noise/2, 1+noise/25, n) * radius
    X, Y = center[0]+R*np.cos(T),  center[1]+R*np.sin(T)
    X = np.r_[X,X[0]]
    Y = np.r_[Y,Y[0]]
    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    TCK, U = interpolate.splprep([X, Y], s=0, per=True)
    # evaluate the spline fits for 1000 evenly spaced distance values
    Xi, Yi = interpolate.splev(np.linspace(0, 1, 1000), TCK)
    
    verts = np.dstack([Xi,Yi]).reshape(len(Xi),2)
    codes = [Path.MOVETO,] + [Path.LINETO,]*(len(verts)-2) + [Path.LINETO,]
    path = Path(verts, codes)
    # return Xi, Yi
    return X, Y, path

    
# Parameters
# ----------
np.random.seed(1)

n = 1024
xmin, xmax = 0, n
ymin, ymax = 0, n
n_cones = 25
n_rods = 2500
n_iter_cones = 15
n_iter_rods = 30
cones_radius = 30
force = 0


# Compute cones locations
# -----------------------
x0, y0 = (xmin+xmax)/2, (ymin+ymax)/2
X, Y = np.meshgrid(np.linspace(xmin, xmax, n, endpoint=False),
                   np.linspace(ymin, ymax, n, endpoint=False))
C = np.sqrt((X-x0)*(X-x0)+(Y-y0)*(Y-y0))
C = normalize(C)
density = 1-np.power(C,0.5)
density_P = density.cumsum(axis=1)
density_Q = density_P.cumsum(axis=1)
points = initialization(n_cones, density)
cones_density = density

if force:
    print("Generating cones locations (n=%d)" % len(points))
    for i in tqdm.trange(n_iter_cones):
        regions, points = voronoi.centroids(points, density, density_P, density_Q)
    cones = points
    np.save("data/cones.npy", cones)
    np.save("data/cones_density.npy", cones_density)
    cones_radii = cones_radius * np.random.uniform(0.9,1.1,len(cones))
    np.save("data/cones_radii.npy", cones_radii)
else:
    cones = np.load("data/cones.npy")
    cones_radii = np.load("data/cones_radii.npy")
    cones_density = np.load("data/cones_density.npy")
    print("Loading cones locations and radii (n=%d)" % len(cones))


# Compute rods locations
# ----------------------
if force:
    density = np.zeros((n,n))
    density[:] = np.linspace(0.00,0.5,n)
    
    for i,(x,y) in enumerate(points):
        C = np.sqrt((X-x)*(X-x)+(Y-y)*(Y-y))
        density[C < cones_radii[i]] = 1
        
    density = 1-normalize(density)
    density_P = density.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)
    rods_density = density
    points = initialization(n_rods, density)
    print("Generating rods locations (n=%d)" % len(points))
    for i in tqdm.trange(n_iter_rods):
        regions, points = voronoi.centroids(points, density, density_P, density_Q)
    rods = points
    np.save("data/rods.npy", rods)
    np.save("data/rods_density.npy", rods_density)
else:
    rods = np.load("data/rods.npy")
    rods_density = np.load("data/rods_density.npy")
    print("Loading rods locations (n=%d)" % len(rods))


# Display
# -------
plt.figure(figsize=(9,6))

ax = plt.subplot2grid((2,3), (0, 0), aspect=1)

ax.imshow(cones_density, extent=[xmin, xmax, ymin, ymax],
           cmap=plt.get_cmap("gray"), origin="lower")
ax.text(24, ymax-24, "A", color="white", weight="bold", va="top", fontsize=16)
ax.text(12, 12, "Bitmap (1024x1024)", color="white", va="bottom", fontsize=8)
ax.set_xticks([])
ax.set_yticks([])


ax = plt.subplot2grid((2,3), (1, 0), aspect=1)
ax.imshow(rods_density, extent=[xmin, xmax, ymin, ymax],
           cmap=plt.get_cmap("gray"), origin="lower")
ax.text(24, ymax-24, "B", color="black", weight="bold", va="top", fontsize=16)
ax.text(12, 12, "Bitmap (1024x1024)", color="black", va="bottom", fontsize=8)

ax.set_xticks([])
ax.set_yticks([])


ax = plt.subplot2grid((2,3), (0, 1), colspan=2, rowspan=2, aspect=1)
facecolor = np.zeros((len(cones),4))
facecolor[:,1] = facecolor[:,2] = np.random.uniform(0.50, 0.65, len(cones))
facecolor[:,0] = facecolor[:,3] = 1


for P in cones:
   X, Y, path = blob(P, 1.15*cones_radius)
   patch = PathPatch(path, edgecolor="black", facecolor="white")
   ax.add_patch(patch)

#plt.scatter(cones[:,0], cones[:,1], s=cones_radii*20, zorder=20,
#            edgecolor="black", facecolor="white", linewidth=1.25)
#plt.scatter(rods[:,0], rods[:,1], s=2.5,
#            edgecolor="none", facecolor="black")

points = np.append(cones, rods).reshape(len(cones)+len(rods), 2)
# points = rods
patches = []
regions, vertices = voronoi_finite_polygons_2d(points)

facecolor = np.zeros((len(regions),4))

for i,region in enumerate(regions):
    patches.append(Polygon(vertices[region]))
    a = np.random.uniform(0.75,1.00)
    facecolor[i] = a,a,a,1


#plt.scatter(points[:,0], points[:,1], s=2.5, zorder=10,
#            edgecolor="black", facecolor="white", linewidth=.25)


# D = scipy.spatial.distance.cdist(cones,rods)
# D.sort(axis=0)
# D = D[0]
# D = (D-D.min())/(D.max()-D.min())
# facecolor = np.zeros((len(patches),4))
# facecolor[:,0] = D
# facecolor[:,1] = D
# facecolor[:,2] = .75+.25*D
# facecolor[:,3] = .5
# facecolor[:] = 1,1,1,1
collection = PatchCollection(patches,
                             facecolor=facecolor, edgecolor="black", linewidth=0.25)
ax.add_collection(collection)

ax.set_xlim(xmin, xmax)
ax.set_xticks([])
ax.set_ylim(ymin, ymax)
ax.set_yticks([])
ax.text(24, ymax-24, "C", color="black", weight="bold", va="top", fontsize=24, zorder=100)

plt.tight_layout()
# plt.savefig("figures/figure-retina.pdf")
plt.show()
