""" ML_ModeSolve.py
This module contains various functions related to internal wave structure in a mixed layer
cf. [D'Asaro 1978]
In particular: 
 - routine to solve the eigenproblem associated with hydrostatic internal waves in the presence of a mixed layer (discontinuity in the density)
 - functions that returns the w and phi structures of the corresponding modes
 - D'Asaro 1978's approx for surf/interior HKE ratio
 NJAL October 2018 """

from __future__ import division
import numpy as np
from cheb import cheb
from scipy.linalg import eig
from scipy.integrate import cumtrapz
import scipy.interpolate as itp
import scipy.optimize as opt 
from netCDF4 import Dataset
import sys
if '/home/lahaye/Coding/Python_ROMS_Modules/lahaye/' not in sys.path:
    sys.path.append('/home/lahaye/Coding/Python_ROMS_Modules/lahaye/')
from comp_zlevs import zlev_w, zlev_rho
if '/home/lahaye/Coding/Py3_ROMS_Modules' not in sys.path:
    sys.path.append('/home/lahaye/Coding/Py3_ROMS_Modules')
from R_tools import vinterp
if "/home/lahaye/Coding/python_science" not in sys.path:
    sys.path.append("/home/lahaye/Coding/python_science")
from SL_chebsolve import SL_chebsolve
    
### Tools
def get_strat(bvfm, zinterp, zin, zml, dml=20, zpyc=-200, Hbt=3000, Nsqml=0, coef_realstrat=1., sm=0.):

    izml = abs(zinterp + 2*dml).argmin() # ad hoc, twice base of ML
    izpyc = izml #abs(zinterp - zpyc).argmin() # +:- permanent pycnocline
    izbt = abs(zinterp + Hbt).argmin()   # domain considered
    Nsqml *= np.ones(len(zml))
    #print(izpyc, izbt)
    #print(zinterp[izpyc], zinterp[izbt], zinterp[izml])
    bvf_int = np.trapz(bvfm[izbt:izpyc], zinterp[izbt:izpyc])/(zinterp[izpyc-1]-zinterp[izbt])

    Nsqin = itp.UnivariateSpline(zinterp, bvfm, s=sm)(zin)
    Nsqin = (1-coef_realstrat)*bvf_int + coef_realstrat*Nsqin
    return Nsqin, Nsqml, bvf_int

def get_strat_lf(path_strat, zin, zml, dml=20, zpyc=-200, Hbt=3000, Nsqml=0):
    nc = Dataset(path_strat, "r")
    hc, Cs_r, Cs_w = nc.hc, nc.Cs_r, nc.Cs_w
    bvfm = nc.variables ['bvf_avg'][:]
    bvfm = np.r_[bvfm[:1,...], 0.5*(bvfm[1:,...]+bvfm[:-1,...]), bvfm[-1:,...]]
    topo = nc.variables['h'][:]
    nc.close()
    zr = zlev_rho(topo, np.zeros(topo.shape), hc, Cs_r)
    zw = zlev_w(topo, np.zeros(topo.shape), hc, Cs_w).squeeze()
    zz = np.r_[zin, zml]
    zinterp = np.unique(zz)
    bvfm = np.nanmean(vinterp(np.moveaxis(bvfm, 0, 2), zinterp, zr, zw, interp_sfc=0) \
                        , axis=(0,1))
    Nsqin, Nsqml, bvf_int = get_strat(bvfm, zinterp, zin, zml, dml, zpyc, Hbt \
                                  , Nsqml, 1., sm=0) 
    return Nsqin, Nsqml, bvf_int

def sym_sqrt(data):
    """ Symmetric square root: returns -sqrt(x) if x<0 """
    return np.sign(data)*np.sqrt(abs(data))

def fasa(kks, dml=-20, Nsqin=1e-5, gred=1e-2):
    """ D'Asaro 1978 ratio (N=0 in ML, density discontinuity at z=-dml (with amplitude propto gred), interior stratif Nsqin """
    return 1./(Nsqin*kks**2*dml**2 + (1-kks**2*gred*dml)**2)

### Full EVP solving
def get_ModAmpRat_old(path_strat, Hbt=-3000, dml=-20, gred=1e-2, zref=-500, zpyc=-200, Nsqml=0., coef_realstrat=1.):
    """ Amplitude ratio from EVP numerical solution --- Big Wrapper"""
    Nz = 120 # WARNING function parameter
    zin, Din, zml, Dml, zz, Dz = ML_solver_grid(Nz, Hbt, dml)
    Nsqin, Nsqml, bvf_int = get_strat_lf(path_strat, zin, zml, dml, zpyc, Hbt, Nsqml, coef_realstrat=1.)   
    izref = abs(zref-zin).argmin()
    Nin, Nml = len(zin), len(zml)

    # solve eigenproblem
    lom, wo, do = ML_solver(zin, Din, Nsqin, zml, Dml, Nsqml, gred)
    # normalize, such that do=1 at bottom
    wo /= do[:1,:]
    do /= do[:1,:]
    # compute WKBJ amplitude below zpyc
    Nin = np.where(zin<=zpyc)[0][-1]
    amp = np.sqrt(wo[:Nin,:]**2*np.sqrt(Nsqin)[:Nin,None] \
             + (do[:Nin,:]**2/np.sqrt(Nsqin)[:Nin,None])).mean(axis=0)
    
    ratio = (do[-1,:])**2 / np.sqrt(Nsqin[izref]/Nsqin[0])
    return (lom, ratio)

def get_ModAmpRat(path_strat, Hbt=-3000, dml=-20, gred=1e-2, zref=-500, zpyc=-200, Nsqml=0., coef_realstrat=1.):
    """ same as get_ModAmpRat but read in low-pass strat files"""
    Nz = 120 # WARNING function parameter
    zin, Din, zml, Dml, zz, Dz = ML_solver_grid(Nz, Hbt, dml)
    Nsqin, Nsqml, bvf_int = get_strat_lf(path_strat, zin, zml, dml, zpyc, Hbt, Nsqml)   
    izref = abs(zref-zin).argmin()
    Nin, Nml = len(zin), len(zml)

    # solve eigenproblem
    if dml > 0: 
        lom, wo, do = ML_solver(zin, Din, Nsqin, zml, Dml, Nsqml, gred)
    else:
        (wo, do), lom, _ = SL_chebsolve(Nsqin, zin, Nz=Nz)
        do /= -lom[None,:].real
    # normalize, such that do=1 at bottom
    wo /= do[:1,:]
    do /= do[:1,:]
    # compute WKBJ amplitude below zpyc
    Nin = np.where(zin<=zpyc)[0][-1]
    amp = np.sqrt(wo[:Nin,:]**2*np.sqrt(Nsqin)[:Nin,None] \
             + (do[:Nin,:]**2/np.sqrt(Nsqin)[:Nin,None])).mean(axis=0)
    
    ratio = (do[-1,:])**2 / np.sqrt(Nsqin[izref]/Nsqin[0])
    return (lom, ratio)

def ML_solver_grid(Nz=120, Hbt=3000, dml=20):
    if dml > 0.:
        Nml = max(int(Nz*dml/Hbt), int(Nz/10))
        Dml, zml = cheb(Nml, [-dml, 0])
    else:
        print('no Mixed Layer')
        Nml = 0
        Dml, zml = np.zeros((0,0)), np.zeros((0,))
    Nin = Nz - Nml
    Din, zin = cheb(Nin, [-Hbt, -dml])
    zz = np.r_[zin, zml]
    Dz = np.zeros((Nz,Nz))
    Dz[:Nin,:Nin] = Din
    Dz[Nin:,Nin:] = Dml
    return zin, Din, zml, Dml, zz, Dz

def ML_solver(zin, Din, Nsqin, zml, Dml, Nsqml, gred=1e-2):
    # solve eigenproblem containing a mixed layer (density discontinuity)
    Nin = len(zin)
    Nml = len(zml)
    Nz = Nin + Nml
    nmod = ((Nin+Nml)//4)
    Lin = np.r_[ np.c_[ np.zeros((Nin,Nin)),     -Din ] \
                , np.c_[ Din,              np.zeros((Nin,Nin)) ] ]
    Lml = np.r_[ np.c_[ np.zeros((Nml,Nml)),    -Dml ] \
               , np.c_[ Dml,              np.zeros((Nml,Nml)) ] ]

    LL = np.zeros((2*Nz,2*Nz))
    LL[:2*Nin,:2*Nin] = Lin
    LL[2*Nin:,2*Nin:] = Lml

    AA = np.diag(np.r_[np.ones(Nin), Nsqin, np.ones(Nml), Nsqml])

    # B.C.
    # w = 0 at z=-Hb, 0
    LL[Nin,:] = np.r_[ np.zeros(Nin), 1, np.zeros(Nin-1+2*Nml) ]
    LL[-1,:] = np.r_[ np.zeros(2*Nz-1), 1 ]
    AA[Nin,Nin] = AA[-1,-1] = 0
    # continuity of w
    LL[2*Nin-1,:] = np.r_[ np.zeros(2*Nin-1), 1, np.zeros(Nml), -1, np.zeros(Nml-1) ]
    AA[2*Nin-1,2*Nin-1] = 0
    # jump in dw/dz
    LL[2*Nin+Nml,:] = np.r_[ np.zeros(Nin-1), 1, np.zeros(Nin), -1, np.zeros(2*Nml-1) ]
    AA[2*Nin+Nml,2*Nin+Nml] = -gred

    lam, vect = eig(LL, AA)
    lom = lam.copy()
    # clean that shit
    inds, = np.where( (np.isfinite(lam)) & (abs(lam.real)<1e3) & (abs(lam.imag)<1e-6) & (lam.real>1e-10) )
    lam, vect = lam[inds], vect[:,inds]
    inds = lam.real.argsort()[:2*nmod]
    vect = vect[:,inds]

    lam = lam[inds]
    ww = np.r_[vect[Nin:2*Nin,:], vect[-Nml:,:]]
    dw = np.r_[vect[:Nin,:], vect[2*Nin:2*Nin+Nml,:]]
    
    return lam, ww, dw


#####################################################################################################
#####  ---  WKBJ functions: not that these are actual w, dw/dz (i.e. does not match wo, do)  ---  #####

def w_ml(z, Nml=0, k=None, dml=None):
    """ z[-1] = 0 // normalized such that w_ml(-dml)=1  ; Nml constant only"""
    if dml is None:
        if isinstance(z, np.ndarray): 
            dml = z[0]
        else:
            dml = -1
    if np.isclose(Nml, 0):
        return z/dml
    elif Nml > 0:
        return np.sin(k*Nml*z)/np.sin(k*Nml*dml)
    elif Nml < 0:
        return np.sinh(k*Nml*z)/np.sinh(k*Nml*dml)
    
def dwdz_ml(z, Nml=0, k=None, dml=None):
    """ z[-1] = 0 // normalized such that w_ml(-dml)=0 ; 
        Nml constant only"""
    if dml is None:
        if isinstance(z, np.ndarray): 
            dml = z[0]
        else:
            dml = -1.
    if not isinstance(z, np.ndarray): z = np.array([z])
    if isinstance(k, np.ndarray) and k.ndim == 1:
        k = k[None,:]
        shapout = (len(z),len(k))
    else: shapout = z.shape
    if np.isclose(Nml, 0):
        return np.ones(shapout)/dml
    elif Nml > 0:
        return k * Nml * np.cos(k*Nml*z[:,None])/np.sin(k*Nml*dml)
    elif Nml < 0:
        return np.cosh(k*Nml*z[:,None])/np.sinh(k*Nml*dml)
    
def w_in(zin, Nsqin, k):
    if zin.shape != Nsqin.shape: zin = zin.reshape(Nsqin.shape) #raise ValueError("something wrong")
    mwkb = cumtrapz(np.sqrt(Nsqin), zin, initial=0)
    if isinstance(k, np.ndarray) and k.ndim == 1:
        k = k[None,:]
    return np.sin(k*mwkb[:,None])/(Nsqin[0]*Nsqin[:,None])**(1./4)/k

def dwdz_in(zin, Nsqin, k):
    if zin.shape != Nsqin.shape: raise ValueError("something wrong")
    mwkb = cumtrapz(np.sqrt(Nsqin), zin, initial=0)
    if isinstance(k, np.ndarray) and k.ndim == 1:
        k = k[None,:]
    return np.cos(k*mwkb[:,None])*(Nsqin[:,None]/Nsqin[0])**(1./4)

def tr_ek(k, Nml, zin, Nsqin, gred):
    if not isinstance(k, np.ndarray): k = np.array([k])
    wml = w_in(zin, Nsqin, k[None,:])[-1,:]
    zml = zin[-1]
    dwml = dwdz_ml(zml, Nml, k[None,:], dml=zml) * wml
    return (dwml - dwdz_in(zin, Nsqin, k[None,:])[-1,:])/k + gred*wml*k

def get_WKBampRat(path_strat, kks, dml=20, Hbt=3000, zref=-500, zpyc=-100, Nsqml=0, gred=1e-2): 
    """ amplitude ratio from WKBJ estimate --- wrapper
    kks: np.ndarray 1D,  is initial guess for every eigenvalues (typically use results from EVP)
    dml, Hbt: depth of ML, depth of fluid (positive) 
    zref, zpyc: rerference depth (negative) and pycnocline depth (negative)
    Nsqml, gred: BVF in mixed layer (e.g. 0.) and buoyancy gap at bottom of ML
    path_strat: where to read strat.
    """
    Nz = 120
    Nzml = max(round(dml/Hbt*Nz), 10)
    Nzin = Nz - Nzml
    zml = np.linspace(-dml, 0, Nzml)
    zin = np.linspace(-Hbt, -dml, Nzin)
    Nsqin, _, _ = get_strat_lf(path_strat, zin, zml, dml, zpyc, Nsqml)
    Nsqref = itp.interp1d(zin, Nsqin)(zref)
    kzos = np.zeros(kks.shape)
    ff = lambda k:tr_ek(k, sym_sqrt(Nsqml), zin, Nsqin, gred).squeeze()
    for ii,ll in enumerate(kks.real):
        kzos[ii] = opt.fsolve(ff, ll)[0]
    wkb_ratref = ((dwdz_ml(zml, sym_sqrt(Nsqml), kzos) \
           * w_in(zin, Nsqin, kzos)[-1,:])**2).mean(axis=0) / np.sqrt(Nsqref/Nsqin[0])
    return wkb_ratref