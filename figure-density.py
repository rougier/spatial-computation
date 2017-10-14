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
from lxml import etree

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


matplotlib.rc('xtick.major', size=10)

# SVG file where everything is defined
svg_filename = "data/gradient-1024x256.svg"
svg_tree = etree.parse(svg_filename).getroot()

# Read dimensions from svg file directly
xmin, xmax = 0, int(svg_tree.get("width"))
ymin, ymax = 0, int(svg_tree.get("height"))


P1 = np.load("data/gradient-1024x256-stipple-1000.npy")
P2 = np.load("data/gradient-1024x256-stipple-2500.npy")
P3 = np.load("data/gradient-1024x256-stipple-5000.npy")
P4 = np.load("data/gradient-1024x256-stipple-10000.npy")


def plot(ax, P, linewidth = 0.5, size = 25):
    reference = 1000
    ratio = reference/len(P)
    size      = max(ratio * size, 3.0)
    linewidth = max(ratio * linewidth, 0.25)
    ax.scatter(P[:,0], P[:,1], facecolor="white", edgecolor="black",
               s=size, linewidth=linewidth)
    ax.set_ylabel("%d cells" % len(P))
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.linspace(xmin,xmax,5, endpoint=True))
    ax.set_xticklabels(["",]*5)

    X = np.linspace(xmin,xmax,5, endpoint=True)
    for x0,x1 in zip(X[:-1], X[1:]):
        n = np.logical_and(P[:,0] >= x0, P[:,0] < x1).sum()
        ratio = 100*n/len(P)
        ax.text((x1+x0)/2, -6, "%.2f%% (n=%d)" % (ratio, n), fontsize=8,
                ha="center", va="top", clip_on=False)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])

plt.figure(figsize=(10,10))

ax = plt.subplot(4,1,1, aspect=1)
plot(ax, P1)

ax = plt.subplot(4,1,2, aspect=1)
plot(ax, P2)

ax = plt.subplot(4,1,3, aspect=1)
plot(ax, P3)

ax = plt.subplot(4,1,4, aspect=1)
plot(ax, P4)


plt.tight_layout()
# plt.savefig("figures/figure-density.pdf")
plt.show()
