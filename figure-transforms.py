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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def no_transform(X, Y):
    return X, Y

def polar_transform(X, Y):
    R = (X+1)/2
    T = -(Y+1)*np.pi
    U = R * np.cos(T)
    V = R * np.sin(T)
    return U, V

def elliptic_transform(X, Y):
    # Analytical Methods for Squaring the Disc
    # Chamberlain Fong, arXiv, 2017
    # https://arxiv.org/pdf/1509.06344.pdf
    
    U = X * np.sqrt(1 - 0.5*Y*Y)
    V = Y * np.sqrt(1 - 0.5*X*X)

    # V = 0.75*V - .25*U*U
    # V = np.where(V<0, .75*V, V)
    # U = np.where(U<0, .75*U, U)
    return U, V


def stretching_transform (X, Y):
    X2 = X*X
    Y2 = Y*Y
    S = np.sqrt(X2+Y2)
    U,V = np.zeros_like(X), np.zeros_like(Y)

    U = np.where(X2 >= Y2, np.sign(X)*(X2 )/S, 0)
    U = np.where(X2 <  Y2, np.sign(Y)*(X*Y)/S, U)
    V = np.where(X2 >= Y2, np.sign(X)*(X*Y)/S, 0)
    V = np.where(X2 <  Y2, np.sign(Y)*(Y2 )/S, V)

    return U,V

def concentric_transform (X, Y):
    # A Low Distortion Map Between Disk and Square,
    # Peter Shirley and Kenneth Chiu, Journal of Graphics Tools, 1997
    # https://mediatech.aalto.fi/~jaakko/T111-5310/K2013/JGT-97.pdf
    def _transform(a,b):
        if a > -b:
            if a > b:
                r = a
                phi = (np.pi/4) * (b/a)
            else:
                r = b
                phi = (np.pi/4) * (2 - (a/b))
        else: 
            if a < b:
                r = -a
                phi = (np.pi/4) * (4 + (b/a))
            else:
                r = -b
                if b != 0:
                    phi = (np.pi/4) * (6 - (a/b))
                else:
                    phi = 0
        u = r * np.cos(phi)
        v = r * np.sin(phi)
        return u, v

    U,V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(len(X)):
        U[i], V[i] = _transform(X[i], Y[i])

    # V = np.where(V<0, .75*V, V)
    # U = np.where(U<0, .75*U, U)
    return U, V


def square(x, y, dx, dy, transform=no_transform):
    n = 10
    Xi = x + np.linspace(0,dx,n)
    X0 = x + np.zeros(n)
    X1 = x + dx*np.ones(n)
    Yi = y + np.linspace(0,dy,n)
    Y0 = y + np.zeros(n)
    Y1 = y + dy*np.ones(n)
    X = np.concatenate((X0,Xi,X1,Xi[::-1]))
    Y = np.concatenate((Yi,Y1,Yi[::-1],Y0))
    P = np.stack(transform(X,Y)).T
    return P

def circle(x, y, r, transform=no_transform):
    T = np.linspace(0,2*np.pi,100)
    X = x + r*np.cos(T)
    Y = y + r*np.sin(T)
    P = np.stack(transform(X,Y)).T
    return P



def draw(transform=None):

    n = 21
    X = np.linspace(-1,1,10*n)
    for y in np.linspace(-1,1,n):
        Y = y*np.ones(len(X))
        U, V = transform(X,Y)
        if y in [-1,1]:
            plt.plot(U,V, color='black', linewidth=1.5)
        elif y == 0:
            plt.plot(U,V, color='black', linewidth=1.0)
        else:
            plt.plot(U,V, color='black', linewidth=0.25)

    Y = np.linspace(-1,1,10*n)
    for x in np.linspace(-1,1,n):
        X = x*np.ones(len(Y))
        U, V = transform(X,Y)
        if x in [-1,1]:
            plt.plot(U,V, color='black', linewidth=1.5)
        elif x == 0:
            plt.plot(U,V, color='black', linewidth=1.0)
        else:
            plt.plot(U,V, color='black', linewidth=0.25)

    patches = []
    p = True
    for x in np.linspace(-1, 1, n-1, endpoint=False):
        for y in np.linspace(-1, 1, n-1, endpoint=False):
            if p:
                P = square(x, y, 1/10, 1/10, transform)
                polygon = Polygon(P, False)
                patches.append(polygon)
                collection = PatchCollection(patches, color='.85')
            p = not p
        p = not p
    ax.add_collection(collection)

    patches = []

    P = circle(-0.3,+0.6, 0.1, transform)
    polygon = Polygon(P, True)
    patches.append(polygon)
    collection = PatchCollection(patches, linewidth=1.0,
                                 edgecolor='black', facecolor='none')

    P = circle(-0.3,+0.6, 0.2, transform)
    polygon = Polygon(P, True)
    patches.append(polygon)
    collection = PatchCollection(patches, linewidth=1.0,
                                 edgecolor='black', facecolor='none')

    P = circle(-0.3,+0.6, 0.3, transform)
    polygon = Polygon(P, True)
    patches.append(polygon)
    collection = PatchCollection(patches, linewidth=1.0,
                                 edgecolor='black', facecolor='none')


    ax.add_collection(collection)


plt.figure(figsize=(15,4.25))

ax = plt.subplot(1,4,1, aspect=1, frameon=False)
draw(transform = no_transform)
plt.xlim(-1.05, +1.05)
plt.ylim(-1.05, +1.05)
plt.xticks([]), plt.yticks([])
plt.title("A - No transform")

ax = plt.subplot(1,4,2, aspect=1, frameon=False)
draw(transform = elliptic_transform)
plt.xlim(-1.05, +1.05), plt.ylim(-1.05, +1.05)
plt.xticks([]), plt.yticks([])
plt.title("B - Elliptic")

ax = plt.subplot(1,4,3, aspect=1, frameon=False)
draw(transform = concentric_transform)
# draw(transform = stretching_transform)
plt.xlim(-1.05, +1.05), plt.ylim(-1.05, +1.05)
plt.xticks([]), plt.yticks([])
plt.title("C - Concentric")

ax = plt.subplot(1,4,4, aspect=1, frameon=False)
draw(transform = polar_transform)
plt.xlim(-1.05, +1.05), plt.ylim(-1.05, +1.05)
plt.xticks([]), plt.yticks([])
plt.title("D - Polar")

plt.tight_layout()
# plt.savefig("figures/figure-transform.pdf")
plt.show()
