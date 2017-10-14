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
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import Polygon
from conformalmapping import *


def cartesian_axis(radius, transform=None):
    lines = []
    A = np.zeros(256)
    B = np.linspace(-radius, radius, 256)
    lines.append(transform(np.dstack([A,B]).squeeze()))
    lines.append(transform(np.dstack([B,A]).squeeze()))
    return lines

def cartesian_grid(radius, n=17, transform=None):
    lines = []
    for a in np.linspace(-radius, radius, n):
        b = np.sqrt(radius*radius-a*a)
        A = a*np.ones(256)
        B = np.linspace(-b, b, 256)
        lines.append(transform(np.dstack([A,B]).squeeze()))
        lines.append(transform(np.dstack([B,A]).squeeze()))
    return lines

def cartesian_checker(radius, n, transform=None):

    # Enclosing disk with enough points
    T = np.linspace(0,2*np.pi, 2*360)
    X = radius*np.cos(T)
    Y = radius*np.sin(T)
    C = Polygon(np.dstack([X,Y]).squeeze())

    paths = []
    x_index, y_index = 0, 0
    for j in range(n):
        for i in range(n):
            index = (x_index+y_index) % 2
            x0 = -radius +     i*(2*radius/n)
            x1 = -radius + (i+1)*(2*radius/n)
            y0 = -radius +     j*(2*radius/n)
            y1 = -radius + (j+1)*(2*radius/n)

            # Build a box with enough points on each side (p=64)
            p = 64
            P = np.zeros((4*p,2))
            P[:p,0] = x0
            P[:p,1] = np.linspace(y0,y1,p,endpoint=False)
            P[p:2*p,0] = np.linspace(x0,x1,p,endpoint=False)
            P[p:2*p,1] = y1
            P[2*p:3*p,0] = x1
            P[2*p:3*p,1] = np.linspace(y1,y0,p,endpoint=False)
            P[3*p:,0] = np.linspace(x1,x0,p,endpoint=False)
            P[3*p:,1] = y0
            B = Polygon(P)

            # Compute the intersection of the box and the circle
            I = C.intersection(B)

            # If they interesect and the color is "on"
            if isinstance(I,Polygon) and index:
                verts = transform(np.array(I.exterior.coords))
                codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-1)
                paths.append( Path(verts, codes))
            x_index += 1
        y_index += 1
        x_index = 0
    return paths


def polar_axis(radius, transform=None):
    lines = []
    A = np.zeros(256)
    B = np.linspace(-radius, radius, 256)
    lines.append(transform(np.dstack([A,B]).squeeze()))
    lines.append(transform(np.dstack([B,A]).squeeze()))
    return lines

def polar_grid(radius, n, transform=None):
    
    lines = []    
    p = 512
    R = np.linspace(0, radius, p)
    for T in np.linspace(0, 2*np.pi, 2*n, endpoint=False):
        X = R * np.cos(T)
        Y = R * np.sin(T)
        lines.append(transform(np.dstack([X,Y]).squeeze()))
    T = np.linspace(0, 2*np.pi, p)
    for R in np.linspace(0, radius, n+1, endpoint=False)[1:]:
        X = R * np.cos(T)
        Y = R * np.sin(T)
        lines.append(transform(np.dstack([X,Y]).squeeze()))
    return lines
    

def polar_checker(radius, n, transform=None):
    
    # Enclosing disk with enough points
    T = np.linspace(0,2*np.pi, 2*360)
    X = radius*np.cos(T)
    Y = radius*np.sin(T)
    C = Polygon(np.dstack([X,Y]).squeeze())

    paths = []
    x_index, y_index = 0, 0

    for T in np.linspace(0, 2*np.pi, 2*n, endpoint=False):
        for R in np.linspace(0, radius, n+1, endpoint=False)[1:]:
            index = (x_index+y_index) % 2

            p = 128
            P = np.zeros((4*p,2))

            r = np.linspace(R, R + radius/(n+1), p)
            t = T
            P[:p,0] = r*np.cos(t)
            P[:p,1] = r*np.sin(t)

            r = R+radius/(n+1)
            t = np.linspace(T, T+(2*np.pi)/(2*n), p)
            P[p:2*p,0] = r*np.cos(t)
            P[p:2*p,1] = r*np.sin(t)
                            
            r = np.linspace(R + radius/(n+1), R, p)
            t = T+(2*np.pi)/(2*n)
            P[2*p:3*p,0] = r*np.cos(t)
            P[2*p:3*p,1] = r*np.sin(t)

            r = R
            t = np.linspace(T+(2*np.pi)/(2*n), T, p)
            P[3*p:,0] = r*np.cos(t)
            P[3*p:,1] = r*np.sin(t)
                            
            if index:
                verts = transform(P) 
                codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-1)
                paths.append( Path(verts, codes))
            x_index += 1
            
        y_index += 1
        x_index = 0
    return paths


def normalize(V):
    V = np.asarray(V)
    Vmin = abs(min(V.real.min(), V.imag.min()))
    Vmax = abs(max(V.real.max(), V.imag.max()))
    return V/max(Vmin,Vmax)



blob1 = normalize([
    0.2398 + 0.6023j,  0.3567 + 1.0819j,  0.2632 + 1.5965j, -0.5205 + 1.7485j,
   -1.0585 + 1.1170j, -1.0702 + 0.5088j, -0.5906 + 0.0994j, -0.7778 - 0.4269j,
   -1.2924 - 0.6140j, -1.4561 - 1.2456j, -0.5439 - 1.3509j,  0.2515 - 1.0702j,
    0.3099 - 0.6023j,  0.7427 - 0.5906j,  1.1053 - 0.1813j,  1.2807 + 0.3567j])
blob2 = normalize([
     0.5896 + 1.2486j, -0.1426 + 1.5954j, -0.9133 + 1.1561j,
    -0.8465 + 0.3536j, -1.1116 - 0.2398j, -1.2695 - 0.9643j,
    -0.5660 - 1.1075j,  0.2013 - 0.7552j,  0.8362 - 0.9634j,
     1.5838 - 0.7013j,  1.3141 + 0.4008j,  0.8474 + 0.7291j])
n = 3
r = 1/np.sqrt(2)
blob3 = (((np.ones(n) + 1j*np.linspace(-1,+1,n,endpoint=False))*r).tolist() +
        ((np.linspace(+1,-1,n,endpoint=False) + 1j*np.ones(n))*r).tolist() +
        ((-np.ones(n) + 1j*np.linspace(1,-1,n,endpoint=False))*r).tolist() +
        ((np.linspace(-1,+1,n,endpoint=False) - 1j*np.ones(n))*r).tolist())


# blob1 = Splinep.from_complex_list(blob1)
# blob2 = Splinep.from_complex_list(blob2)
# blob3 = Splinep.from_complex_list(blob3)

# sm1 = SzMap(blob1, 0)
# sm2 = SzMap(blob2, 0)
# sm3 = SzMap(blob3, 0)
# def transform_conformal_1(V):
#     V_ = np.empty_like(V)
#     Vc = sm.apply(V[:,0]+1j*V[:,1])
#     V_[:,0] = Vc.real
#     V_[:,1] = Vc.imag
#     return V_

# def transform_conformal_2(V):
#     V_ = np.empty_like(V)
#     Vc = sm.apply(V[:,0]+1j*V[:,1])
#     V_[:,0] = Vc.real
#     V_[:,1] = Vc.imag
#     return V_

# def transform_conformal_3(V):
#     V_ = np.empty_like(V)
#     Vc = sm.apply(V[:,0]+1j*V[:,1])
#     V_[:,0] = Vc.real
#     V_[:,1] = Vc.imag
#     return V_

# def transform_identity(V):
#     return V.copy()



def circle(center, r):
    T = np.linspace(0,2*np.pi,300)
    X = center[0] + r*np.cos(T)
    Y = center[1] + r*np.sin(T)
    return np.dstack([X,Y]).reshape(len(T),2)


def draw(ax, blob=None, mode=0):

    # Transform
    if blob is not None:
        blob = Splinep.from_complex_list(blob)
        sm = SzMap(blob, 0)
        def transform(V):
            V_ = np.empty_like(V)
            Vc = sm.apply(V[:,0]+1j*V[:,1])
            V_[:,0] = Vc.real
            V_[:,1] = Vc.imag
            return V_
        Z = blob.point(np.linspace(0.0, 1.0, 360))
        X,Y = Z.real, Z.imag
    else:
        def transform(V):
            return V.copy()    
        T = np.linspace(0,2*np.pi, 2*360)
        X,Y = np.cos(T), np.sin(T)
        
    ax.plot(X, Y, linewidth=1.5, color="black")

    
    if mode:
#        for line in cartesian_grid(1, 32, transform):
#            X, Y = line[:,0], line[:,1]
#            ax.plot(X, Y, linewidth=0.5, color="black", alpha=0.25)
        for path in cartesian_checker(1, 32-1, transform):
            patch = patches.PathPatch(path, facecolor="#BDBEDA", alpha=1.0,
                                      edgecolor="none", linewidth=0)
            ax.add_patch(patch)
        for i,line in enumerate(cartesian_axis(1, transform)):
            X, Y = line[:,0], line[:,1]
            if i == 0:
                ax.plot(X, Y, linestyle="--", linewidth=1.5, color="blue")
            else:
                ax.plot(X, Y, linestyle="--", linewidth=1.5, color="red")


    else:
#        for line in polar_grid(1, 16, transform):
#            X, Y = line[:,0], line[:,1]
#            ax.plot(X, Y, linewidth=0.5, color="black", alpha=0.5)

        for path in polar_checker(1, 16, transform):
            patch = patches.PathPatch(path, facecolor="#BDBEDA", alpha=1.0,
                                      edgecolor="none", linewidth=0)
            ax.add_patch(patch)
        for i,line in enumerate(polar_axis(1, transform)):
            X, Y = line[:,0], line[:,1]
            if i == 0:
                ax.plot(X, Y, linestyle="--", linewidth=1.5, color="blue")
            else:
                ax.plot(X, Y, linestyle="--", linewidth=1.5, color="red")

    center = 0.4, 0.4
    P = transform(circle(center, 0.1))
    ax.plot(P[:,0], P[:,1], linewidth=1, color="black")
    
    P = transform(circle(center, 0.2))
    ax.plot(P[:,0], P[:,1], linewidth=1, color="black")
    
    P = transform(circle(center, 0.3))
    ax.plot(P[:,0], P[:,1], linewidth=1, color="black")
                
    ax.set_xlim(-1.1,1.1), ax.set_ylim(-1.1,1.1)
    ax.set_xticks([]), ax.set_yticks([])




fig = plt.figure(figsize=(16,8))

ax = plt.subplot(2, 4, 1, aspect=1, frameon=False)
draw(ax, None)
ax = plt.subplot(2, 4, 2, aspect=1, frameon=False)
draw(ax, blob3)
ax = plt.subplot(2, 4, 3, aspect=1, frameon=False)
draw(ax, blob2)
ax = plt.subplot(2, 4, 4, aspect=1, frameon=False)
draw(ax, blob1)

ax = plt.subplot(2, 4, 5, aspect=1, frameon=False)
draw(ax, None, 1)
ax = plt.subplot(2, 4, 6, aspect=1, frameon=False)
draw(ax, blob3, 1)
ax = plt.subplot(2, 4, 7, aspect=1, frameon=False)
draw(ax, blob2, 1)
ax = plt.subplot(2, 4, 8, aspect=1, frameon=False)
draw(ax, blob1, 1)

plt.tight_layout()
# plt.savefig("figure-conformal-maps.pdf")
plt.show()

