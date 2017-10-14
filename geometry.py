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
import scipy.spatial.distance

def signed_distance_segment(P1, P2, P):
    """
    Compute the signed distance from points P to a broken line made of 
    individual segments P1-P2
    """
    X = P[:,0].reshape(len(P),1)
    Y = P[:,1].reshape(len(P),1)    
    X1, Y1 = P1[:,0], P1[:,1]
    X2, Y2 = P2[:,0], P2[:,1]
    dX, dY = X2-X1, Y2-Y1
    A = dY*X - dX*Y + X2*Y1 - Y2*X1
    B = dX*dX + dY*dY
    # D = A/np.sqrt(B)
    D = np.divide(A, np.sqrt(B), out=np.zeros_like(A), where=B!=0)
    # U = ((X-X1)*dX + (Y-Y1)*dY) / B
    U = np.divide((X-X1)*dX+(Y-Y1)*dY, B, out=np.zeros_like(A), where=B!=0)
    D1 = scipy.spatial.distance.cdist(P, P1)
    D2 = scipy.spatial.distance.cdist(P, P2)
    D_ = np.abs(D)
    D_ = np.where(U < 0, D1, D_)
    D_ = np.where(U > 1, D2, D_)
    i = np.arange(len(P))
    j = D_.argmin(axis=1)
    return (D_[i,j] * np.sign(D[i,j]))
    
def signed_distance_polyline(L, P):
    """
    Compute the signed distance from points P to a broken line L made of 
    individual segments
    """

    P1, P2 = L[:-1], L[1:]
    
    X = P[:,0].reshape(len(P),1)
    Y = P[:,1].reshape(len(P),1)    
    X1, Y1 = P1[:,0], P1[:,1]
    X2, Y2 = P2[:,0], P2[:,1]
    dX, dY = X2-X1, Y2-Y1
    A = dY*X - dX*Y + X2*Y1 - Y2*X1
    B = dX*dX + dY*dY
    # D = A/np.sqrt(B)
    D = np.divide(A, np.sqrt(B), out=np.zeros_like(A), where=B!=0)
    # U = ((X-X1)*dX + (Y-Y1)*dY) / B
    U = np.divide((X-X1)*dX+(Y-Y1)*dY, B, out=np.zeros_like(A), where=B!=0)
    D1 = scipy.spatial.distance.cdist(P, P1)
    D2 = scipy.spatial.distance.cdist(P, P2)
    D_ = np.abs(D)
    D_ = np.where(U < 0, D1, D_)
    D_ = np.where(U > 1, D2, D_)
    i = np.arange(len(P))
    j = D_.argmin(axis=1)
    return (D_[i,j] * np.sign(D[i,j]))

