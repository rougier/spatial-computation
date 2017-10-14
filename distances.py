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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def connect(P, method="k-nearest", k=5, radius=1.0):
    """
    Parameters
    ----------

    method : string
      "k-nearest": connect to k nearest neighbors
      "radius":    connect every neighbors within radius distance

    k : int
      Number of neighbors (but self) to connect

    radius : float
    
    """

    n = len(P)
    
    # Angles
    dP = P.reshape(1,n,2) - P.reshape(n,1,2)
    A = np.zeros((n,n))
    for i in range(n):
        A[i] = np.arctan2(dP[i,:,1], dP[i,:,0]) * 180.0 / np.pi
    
    # Distances
    D = np.hypot(dP[...,0], dP[...,1])
    I = np.argsort(D, axis=1)

    
    W = np.zeros((n,n))
    for i in range(n):
        W[i,I[i,1:k+1]] = 1

    """
    # Connections (no self-connection -> 1:k+1)
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
    

filename = "gradient-stipple-1000.npy"
# filename = "gradient-stipple-2500.npy"
# filename = "gradient-stipple-5000.npy"
# filename = "gradient-stipple-10000.npy"

# Position
P = np.load(filename)
n = len(P)
W = connect(P, "k-nearest", k=5)

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

X, Y = P[:,0], P[:,1]
xmin, xmax = 0.00, 1.00
ymin, ymax = 0.00, 0.25
plt.figure(figsize=(11,3))
ax = plt.subplot(1,1,1, aspect=1)
ax.scatter(X, Y, s=sizes, facecolor=facecolors, edgecolor=edgecolors,
           linewidth=linewidths)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xticks([])
ax.set_yticks([])


segments = []
for i in samples:
    src = P[i]
    for j in W[i].nonzero()[0]:
        tgt = P[j]
        segments.append([src, tgt])
c = LineCollection(segments, color="black", linewidth=1.0, zorder=-10)
ax.add_collection(c)


plt.tight_layout()
plt.show()










# # Angles
# Z = P.reshape(1,n,2) - P.reshape(n,1,2)
# A = np.zeros((n,n))
# for i in range(n):
#     A[i] = np.arctan2(Z[i,:,1], Z[i,:,0]) * 180.0 / np.pi
    
# # Distances
# D = scipy.spatial.distance.cdist(P, P)
# I = np.argsort(D, axis=1)


# W = np.zeros((n,n))
# W[I] = 1

# Histogram for distances to the n nearest neighbors
# D = scipy.spatial.distance.cdist(P, P)
# D.sort(axis=0)
# N = D[:,:10].flatten()
# plt.hist(N, bins = 50)
# plt.show()


# X, Y = P[:,0], P[:,1]
# xmin, xmax = 0.00, 1.00
# ymin, ymax = 0.00, 0.25
# plt.figure(figsize=(12,4))
# ax = plt.subplot(1,1,1, aspect=1)
# ax.scatter(X, Y, s=25, facecolor='white', edgecolor='black', linewidth=0.5)
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)
# ax.set_xticks([])
# ax.set_yticks([])
#ax = plt.subplot(2,1,2)
#H, _, _ = np.histogram2d(X, Y, bins=(64,16))
#plt.imshow(H.T, origin="lower", interpolation="bicubic",
#           extent=[xmin, xmax, ymin, ymax])
#ax.set_xlim(xmin, xmax)
#ax.set_ylim(ymin, ymax)
#ax.set_xticks([])
#ax.set_yticks([])


# segments = []
# index = 10
# k = 6
# for index in np.random.randint(0,n,10): #[0,10,20]: #range(n):
#     p0  = P[index]
#     for i in range(k):
#         p1  = P[I[index,i+1]]
#     # p1  = P[i]
#         d = p0-p1
#         #if abs(d[0]) < .1 and d[1] > 0 and D[index,i] < .5:
#         segments.append([p0,p1])
#     # if D[index,i] < .05 and -135 < A[index,i] < -45:
#     #if D[index,i] < .05:
#         #plt.plot([p0[0],p1[0]], [p0[1],p1[1]], color="black", linewidth=.5,
#         #         zorder=-10)
#         # segments.append([p0,p1])

# collection = LineCollection(segments, color="black", linewidth=1.0, zorder=-10)
# ax.add_collection(collection)
# plt.show()




"""
plt.imshow(D)
plt.show()
"""

#Dmax = D.max()
#N = (D < 0.05).sum(axis=0).astype(int)
#plt.hist(N, bins=50)
#plt.show()
#X = X/1024.0
#Y = Y/1024.0


# D += np.eye(len(D))*1e5
#print(N.min(), N.max())
#print(len(N))
#D = (D  > 0) * (D < 0.1*D.max())
#N = D.sum(axis=0)
#print(N.max())
#plt.imshow(D, interpolation="nearest", cmap=plt.cm.gray_r)

