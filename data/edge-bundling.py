import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


def curve(P0,P1):

    x = .25
    y = .15

    length = np.sqrt(((P1-P0)**2).sum())

    
    T = (P1-P0)/length
    O = np.array([-T[1],T[0]])

    V = np.zeros((8,2))
    C = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
         Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    V[0] = P0
    V[1] = P0 + 0.5*x*(P1-P0)
    V[2] = P0 + 0.5*x*(P1-P0) + 0.5*y*O
    V[3] = P0 + 1.0*x*(P1-P0) + 0.5*y*O
    V[4] = P1 + 1.0*x*(P0-P1) + 0.5*y*O
    V[5] = P1 + 0.5*x*(P0-P1) + 0.5*y*O
    V[6] = P1 + 0.5*x*(P0-P1) 
    V[7] = P1

    #plt.scatter(V[:,0], V[:,1], edgecolor="k",facecolor="w",lw=.5,zorder=5,s=10)
    #plt.plot(V[:,0], V[:,1], "--", lw=.5, color="k")
    return Path(V, C)




fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect=1)

n = 10
G0 = (0.25, 0.25) + 0.1*np.random.normal(0, 1, (n,2))
C0 = G0.sum(axis=0)/len(G0)
G1 = G0 + (.5,.5)
C1 = G1.sum(axis=0)/len(G1)

for P0,P1 in zip(G0,G1):
    #P0 = np.array([0,0])
    #P1 = np.array([4,1])
    path = curve(P0,P1)
    patch = patches.PathPatch(path, facecolor='none')
    ax.add_patch(patch)
    
ax.set_xlim(-.1,1), ax.set_ylim(-.1,1)
plt.show()





"""
def projection(P, P0, P1):
    x, y = P
    x0, y0 = P0
    x1, y1 = P1
    l2 = ((P1-P0)**2).sum()
    u = ((x-x0)*(x1-x0)+(y-y0)*(y1-y0))/l2
    x = x0 + u*(x1-x0)
    y = y0 + u*(y1-x0)
    return np.array([x,y])


def attract(P, P0, P1, alpha):
    H = projection(P,P0,P1)

    return P + alpha*(H-P)

    
n = 24
G0 = (0.25, 0.25) + 0.1*np.random.normal(0, 1, (n,2))
C0 = G0.sum(axis=0)/len(G0)
print(C0)


G1 = G0 + (0.5, 0.5) #+ 0.05*np.random.normal(0, 1, (n,2))
C1 = G1.sum(axis=0)/len(G1)
print(C1)

fig = plt.figure(figsize=(6,6))
ax = plt.subplot(1,1,1)

ax.scatter(G0[:,0], G0[:,1], s=25)
ax.scatter(C0[0], C0[1], s=25, edgecolor="black", facecolor="white")

ax.scatter(G1[:,0], G1[:,1], s=25)
ax.scatter(C1[0], C1[1], s=25, edgecolor="black", facecolor="white")

ax.set_xlim(0,1), ax.set_ylim(0,1)


for x in np.linspace(0,1,50):
    P = (1-x)*G0[0] + x * G1[0]
    ax.scatter(P[0],P[1],s=10,color=".5")
    P = attract(P, C0, C1, pow(min(x,1-x),.1))
    ax.scatter(P[0],P[1],s=10,color="k")
"""

