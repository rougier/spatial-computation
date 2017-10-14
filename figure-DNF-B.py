# Dynamic Neural Field simulation
# Copyright (c) 2017 Nicolas P. Rougier
'''
Dynamic neural field
====================

This script implements the numerical integration of dynamic neural field of the
form:
                  
  ∂U(x,t)             ⌠+∞ 
τ ------- = -U(x,t) + ⎮  w(|x-y|).f(U(y,t)).dy + I(x,t) + h
    ∂t                ⌡-∞ 

where U(x,t) is the potential of a neural population at position x and time t
      W(d) is a neighborhood function from ℝ⁺ → ℝ
      f(x) is the firing rate of a single neuron from ℝ → ℝ
      I(x,t) is the input at position x and time t
      h is the resting potential
      τ is the temporal decay of the synapse

References:
    http://www.scholarpedia.org/article/Neural_fields
'''
import numpy as np
import scipy.spatial
import scipy.linalg


def distance(n=40, center=(0,0)):
    xmin, xmax = -1, +1
    ymin, ymax = -1, +1
    x0, y0 = center
    X, Y = np.meshgrid(np.linspace(xmin-x0, xmax-x0, n, endpoint=True),
                       np.linspace(ymin-y0, ymax-y0, n, endpoint=True))
    return np.sqrt(X*X+Y*Y)


def gaussian(D, sigma=1.0):
    return np.exp(-0.5*(D**2/sigma**2))


def convolve1d( Z, K ):
    R = np.convolve(Z, K, 'same')
    i0 = 0
    if R.shape[0] > Z.shape[0]:
        i0 = (R.shape[0]-Z.shape[0])//2 + 1 - Z.shape[0]%2
    i1 = i0 + Z.shape[0]
    return R[i0:i1]


def convolve2d(Z, K, USV = None):
    epsilon = 1e-9
    if USV is None:
        U,S,V = scipy.linalg.svd(K)
        U,S,V = U.astype(K.dtype), S.astype(K.dtype), V.astype(K.dtype)
    else:
        U,S,V = USV
    n = (S > epsilon).sum()
    R = np.zeros(Z.shape)
    for k in range(n):
        Zt = Z * S[k]
        for i in range(Zt.shape[0]):
            Zt[i,:] = convolve1d(Zt[i,:], V[k,::-1])
        for i in range(Zt.shape[1]):
            Zt[:,i] = convolve1d(Zt[:,i], U[::-1,k])
        R += Zt
    return R


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    #  Parameters
    # ----------------------------------------------------------
    seed  = 123 # Seed for the random number generator

    n = 50
    xmin, xmax = -(n-1)/(2*n), +(n-1)/(2*n)
    ymin, ymax = -(n-1)/(2*n), +(n-1)/(2*n)
    X, Y = np.meshgrid(np.linspace(xmin, xmax, n, endpoint=True),
                       np.linspace(ymin, ymax, n, endpoint=True))
    X, Y = X.ravel(), Y.ravel()

    
    P = np.load("data/uniform-1024x1024-stipple-1600.npy")
    # P = np.load("./circular-gradient-1024x1024-stipple-1600.npy")

    pmin, pmax = P.min(), P.max()
    P = (P - pmin)/(pmax-pmin)
    X = xmin + (xmax-xmin)*P[:,0]
    Y = ymin + (ymax-ymin)*P[:,1]

    
    P = np.stack([X,Y], axis=1)
    D = scipy.spatial.distance.cdist(P,P)
    n           = len(P) # Number of neurons
    dt          = 0.10   # Timestep (seconds)
    duration    = 10.0   # Simulation duration (seconds)
    tau         = 0.75   # Time constant (seconds)
    h           = 0.0    # Resting potential
    I_weight    = 0.2   # Weight from input to field
    s        = (40*40)/n # Scaling factor

    sigma_e     = 0.050   # Sigma for excitatory connections
    scale_e     = 0.175*s # Intensity for excitatory connections
    sigma_i     = 0.085   # Sigma for inhibitory connections
    scale_i     = 0.065*s # Intensity for inhibitory connections

    noise       = 0.10   # White noise level
    stim_d      = 100    # Stim discretization level
    stim_n      = 3      # Number of stimulus
    stim_sigma  = 0.1    # Stimulus width
    stim_rho    = 0.65   # Stimulus distance from center
    stim_theta  = 0.0    # Current angular position
    stim_dtheta = 0.0005 # Stimulus angular speed
    def f(x):            # Activation function 
        return np.minimum(np.maximum(x, 0.0), 1.0)


    # Initialization
    # ----------------------------------------------------------
    np.random.seed(1)
    I = np.zeros((100,100))
    Xi = ((I.shape[0]-1)*(X - xmin)/(xmax-xmin)).astype(int)
    Yi = ((I.shape[1]-1)*(Y - ymin)/(ymax-ymin)).astype(int)
    
    U, V = np.zeros(n), np.zeros(n)
    W = scale_e*gaussian(D, sigma=sigma_e) - scale_i*gaussian(D, sigma=sigma_i)

    I[...] = 0.5
    I += np.random.uniform(-noise/2, +noise/2, (100,100))


    # Simulation
    # ----------------------------------------------------------
    def update(frame):
        global I, U, V

        for i in range(10):
            L = np.dot(W,U) 
            V = V + dt/tau*(-V + L + I_weight*I[Yi,Xi] + h)
            U = f(V)

        bins = (32,32)
        H, _, _ = np.histogram2d(Y, X, bins=bins, weights=U, 
                                 range=[[-1, 1], [-1, +1]])
        im_U.set_data(H)
        im_U.set_clim(0,H.max())
        

    # Visualization
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(1,1,1)
#    ax.set_facecolor("k")

    for i in range(1000):
        L = np.dot(W,U) 
        V = V + dt/tau*(-V + L + I_weight*I[Yi,Xi] + h)
        U = f(V)

    bins = (40,40)
    H, _, _ = np.histogram2d(Y, X, bins=bins, weights=U, 
                                 range=[[-1, 1], [-1, +1]])
    im_U = plt.imshow(H, vmin=0, vmax=H.max(), interpolation="bicubic",
                      extent=[-1,+1,-1,+1], origin="lower")
    
    ax.scatter(P[:,0], P[:,1], s=5,
               edgecolor="none", facecolor="white", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    d= 0.01
    ax.set_xlim(xmin-d, xmax+d)
    ax.set_ylim(ymin-d, ymax+d)
    
    ax.text(0.025,+0.975, "B", color="white", ha="left", va="top",
            weight="bold", size=24, transform=ax.transAxes)
    
    # animation = FuncAnimation(fig, update)
    plt.tight_layout()

    # plt.savefig("../data/figure-DNF-B.pdf")
    plt.show()
