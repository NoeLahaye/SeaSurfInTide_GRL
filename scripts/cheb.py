""" cheb.py
This module aims at providing different functions and tools involving chebyshev polynomial
So far only cheb grid and diff. matrix
NJAL October 2018 """
from __future__ import division
import numpy as np

def cheb(N, inter=None):
    N -= 1

    if N==0:
        return None
    xx = np.cos(np.pi*np.arange(N+1)/N)
    cc = np.r_[2, np.ones(N-1), 2]*(-1)**np.arange(N+1)
    X = np.tile(xx[:,None], (1, N+1))
    dX = X - X.T
    D = (cc[:,None]/cc[None,:])/(dX + np.diag(np.ones(N+1)))
    D = D - np.diag(D.sum(axis=1))

    if not inter is None:
        L = inter[1] - inter[0]
        D = -D*2/L
        xx = (xx[::-1] + 1) * L/2. + inter[0]
    return D, xx