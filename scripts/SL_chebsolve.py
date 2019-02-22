import numpy as np
import scipy.linalg as la
import scipy.interpolate as itp
from cheb import cheb

def SL_chebsolve(alsq, zw, Nmod="auto", Nz="auto", grav=0, sm=0, ksplin=3, zbot=None):
    """ solve sturm liouville problem with ev k: w'' + k*alsq*w = 0, w(-H)=w(0)=0
    between zbot (default: zw[0]) and zw[-1] = 0
    wmod and umod (=wmod') are normalized by max value, with u positive at surface
    if grav != 0: free-surface boundary condition. 
    return tuple with (wmod, umod), eigenvalue sqrt(k) and z-cheb
    """
    if Nz=="auto":
        Nz = int(len(zw)*3/2.)
    if Nmod == "auto":
        Nmod = int(Nz/2)
    if zbot is None: zbot = zw[0]    

    Dz, zz = cheb(Nz, [zbot, zw[-1]])
    alsq = itp.UnivariateSpline(zw, alsq, k=ksplin, s=sm)(zz)

    LL = np.r_[ np.c_[  np.diag(np.ones(Nz)), -Dz ] \
               , np.c_[ -Dz,                   np.zeros((Nz,Nz)) ] ]
    AA = np.diag(np.r_[np.zeros(Nz), alsq])
    # BCs
    LL[Nz,:] = 0. # bottom
    LL[-1,:] = 0. # top
    if grav > 0:
        LL[-1,-1] = 0.
        AA[-1,-1] = grav
        LL[-1,Nz-1] = 1.

    lam, vect = la.eig(LL, AA)
    
    lom = lam.copy()
    # clean that stuff
    inds, = np.where( (np.isfinite(lam)) & (abs(lam.real)<1e3) & (abs(lam.imag)<1e-6) & (lam.real>0) )
    lam, vect = lam[inds], vect[:,inds]
    inds = lam.real.argsort()[:Nmod]
    vect = vect[:,inds]/abs(vect[:,inds]).max(axis=0)[None,:]
    lam = lam[inds]

    ww = vect[Nz:,:]
    uu = vect[:Nz,:]
    ww *= np.sign(uu[-1:,:])
    uu *= np.sign(uu[-1:,:])
    
    return (ww, uu), np.sqrt(lam), zz
