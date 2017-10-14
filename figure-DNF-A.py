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
import scipy.linalg
from scipy.ndimage.filters import convolve


def gaussian(n=40, center=(0,0), sigma=0.1):
    xmin, xmax = -1, +1
    ymin, ymax = -1, +1
    x0, y0 = center
    X, Y = np.meshgrid(np.linspace(xmin-x0, xmax-x0, n, endpoint=True),
                       np.linspace(ymin-y0, ymax-y0, n, endpoint=True))
    D = X*X+Y*Y
    return np.exp(-0.5*D/sigma**2)


def convolve1d( Z, K ):
    # return convolve(Z, K, mode='constant')
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
    seed        = 1      # Seed for the random number generator
    n           = 40     # Size of the neural field
    dt          = 0.10   # Timestep (seconds)
    duration    = 10.0   # Simulation duration (seconds)
    tau         = 0.75   # Time constant (seconds)
    h           = 0.0    # Resting potential
    I_weight    = 0.1    # Weight from input to field
    s = (40*40)/(n*n)    # Scaling factor

    sigma_e     = 0.05  # Sigma for excitatory connections
    scale_e     = 0.15*s # Intensity for excitatory connections
    sigma_i     = 0.085  # Sigma for inhibitory connections
    scale_i     = 0.05*s # Intensity for inhibitory connections

    noise       = 0.10   # White noise level
    stim_n      = 3      # Number of stimulus
    stim_sigma  = 0.1    # Stimulus width
    stim_rho    = 0.75   # Stimulus distance from center
    stim_theta  = 0.0    # Current angular position
    stim_dtheta = 0.0005 # Stimulus angular speed
    def f(x):            # Activation function 
        return np.minimum(np.maximum(x, 0.0), 1.0)


    # Initialization
    # ----------------------------------------------------------
    np.random.seed(1)
    I = np.zeros((n,n))
    U, V = np.zeros((n,n)), np.zeros((n,n))
    K = (scale_e*gaussian(2*n+1, sigma=sigma_e) -
         scale_i*gaussian(2*n+1, sigma=sigma_i))
    USV = scipy.linalg.svd(K)

    I[...] = 0.5 + np.random.uniform(-noise/2, +noise/2, (n,n))


    # Simulation
    # ----------------------------------------------------------
    def update(frame):
        global I, U, V, stim_theta, stim_dtheta, noise

        for i in range(10):
            L = convolve2d(U, K, USV)
            V = V + dt/tau*(-V + L + I_weight*I + h)
            U = f(V)

        im_U.set_data(U)
        im_U.set_clim(0,U.max())


    # Visualization
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(1,1,1)


    for i in range(1000):
        L = convolve2d(U, K, USV)
        V = V + dt/tau*(-V + L + I_weight*I + h)
        U = f(V)

    
    im_U = plt.imshow(U, vmin=0, vmax=1, interpolation="bicubic",
                      extent=[-1,+1,-1,+1], origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.025,+0.975, "A", color="white", ha="left", va="top",
            weight="bold", size=24, transform=ax.transAxes)
    plt.tight_layout()

#    plt.savefig("../data/figure-DNF-A.pdf")
    plt.show()
