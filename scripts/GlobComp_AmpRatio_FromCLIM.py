# coding: utf-8
from __future__ import print_function, division
import numpy as np
import scipy.interpolate as itp
from netCDF4 import Dataset
import sys, time
from datetime import datetime
from mpi4py import MPI 
#from cheb import cheb
#from ML_ModeSolve import *
if "/home/lahaye/Coding/python_science" not in sys.path:
    sys.path.append("/home/lahaye/Coding/python_science")
from SL_chebsolve import SL_chebsolve
#from dynmodes_fs import dynmodes_fs as dynmodes
#if "/home/lahaye/Coding/Py3_ROMS_Modules" not in sys.path:
    #sys.path.append("/home/lahaye/Coding/Py3_ROMS_Modules")
#from R_tools import vinterp

# TODO 
# - add description
# - handle number of mode found < nmods
# - remove warnings
# - reduce number of points computed (near the poles)

KRYPTON = "/data0/project/vortex/lahaye/"
RUCHBA = "/net/ruchba/local/tmp/2/lahaye/"
ALPHA = "/net/alpha/exports/sciences/data/"

#####  ---  user-editable parameters  ---  #####
what = "ISAS" # only ISAS is implemented
month = 8

zref = 500
nmods = 10
zpyc = .8*zref

nproc = 8
lon_min, lon_max = -180, 180 # -40, -25 # 
lat_min, lat_max =  -75, 75 # 30, 45 #
substep = 2 # subsampling step

#parameters for resolving SL problem
Nzcal = 80
sm = 1e-12 # smoothing for interpolating spline
ksplin = 2 # interpolating spline degree
indzo = 5 # max z-index to force 0 (discard diurnal layer effects)
grav = 9.81 # put 0 for rigid lid

data_paths =  {"ISAS":ALPHA+"LPO_ISAS/CLIM_ISAS13/"}
topo_paths = {"ISAS":"bathy_GLOBAL05_v6c2.nc"}
filenames = {"ISAS":"ISAS13FD_m{:02d}_BVF2.nc".format(month)}

path_data = data_paths[what]
path_topo = path_data+topo_paths[what]
fname = filenames[what]

fwrite = KRYPTON+"DIAG/SeaSurf_ModAmpRat_m{0:02d}.{1}.nc".format(month,"{}") # .format(ipc)
if grav > 0:
    fwrite = fwrite.replace('Rat_m','Rat_FS_m')

#####  ---  Define functions  ---  #####
def modampratio(nsq, zz, Nzcal=None, grav=0, nmods=10, sm=0, ksplin=2, zref=-500):
    # uses SL_chebsolve for szolving problem using collocation method based on Chebyshev polynomials
    # returns surf/interior modal amplitude ratio and eigenvalue
    # nsq: BVF square
    # zz: depth in ascending order (first point at surface, last point at bottom)
    
    if Nzcal is None: Nzcal = len(zz)
        
    (wo, do), lom, zcheb = SL_chebsolve(nsq[::-1], -zz[::-1], nmods+int(grav>0), Nz=Nzcal, grav=grav, sm=sm, ksplin=ksplin)
    lom = lom.real
    if grav>0:
        if lom[1]>2*lom[0]: # detect barotropic mode
            lom, wo, do = lom[1:], wo[:,1:], do[:,1:]
        elif len(lom)>nmods:
            lom, wo, do = lom[:-1], wo[:,:-1], do[:,:-1]
    do /= -lom[None,:]

    # compute WKBJ amplitude below zpyc
    indz, = np.where(nsq>0)
    if len(indz)==0: 
        print("proc {0}: no pos nsq found in modampratio".format(ipc))
        return None
    zmax, zmin = -max(zpyc,zz[indz.min()]), -zz[indz.max()]
    if (zmin>-zref) | (zmax<-zref): # also includes zmin > zmax
        print("proc {0}: zref not between zmin, zmax:".format(ipc),zmin,zmax)
        return None
    inbz, = np.where((zcheb>zmin) & (zcheb<zmax))
    if len(inbz)==0:
        print("proc {0}: no cheb points available in modampratio".format(ipc))
        print("          zmin, zmax:", zmin, zmax)
        return None
    zchb = zcheb[inbz]
    Nsqin = itp.UnivariateSpline(zz[indz], nsq[indz], k=1, s=sm)(-zchb)
    #indz, = np.where((zcheb<=-zpyc) & (Nsqin>0))
    Nsqin[Nsqin<0] = nsq[indz].min() # limiter to avoid zeros
    izref = abs(zchb+zref).argmin()
    amp = np.sqrt(wo[inbz,:]**2*np.sqrt(Nsqin[:,None]) + (do[inbz,:]**2/np.sqrt(Nsqin[:,None]))).mean(axis=0)

    keint = np.trapz(do*do.conj(), zcheb, axis=0) # TODO use clenshaw-curtis with fixed weight from cheb
    # normalize such that do is 1 (in ampliute) at zref -> ratio is just square of surface value
    norm = amp * Nsqin[izref]**(1/4)
    
    return (do[-1,:]/norm)**2, lom, keint/norm**2, np.sqrt(Nsqin[izref])

def init_netcdf(fwrite, lon, lat_s, topo, indy, i0=0):
    # initialize netCDF files to store results
    ncw = Dataset(fwrite, "w", data_model="NETCDF4_CLASSIC")
    ncw.generated_on = datetime.today().isoformat()
    ncw.generating_script = sys.argv[0]
    ncw.clim_database = what
    ncw.lat_min = lat_min
    ncw.lat_max = lat_max
    ncw.zref = zref
    ncw.zref_expl = "interior reference value for internal wave WKBJ amplitude"
    ncw.subsamp_step = substep

    ny, nx = len(lat_s), len(lon)
        
    for dim,siz in zip(('lon','lat','mode'),(nx,ny,nmods)):
        ncw.createDimension(dim, siz)
    ncw.createVariable("mode","i",("mode",))[:] = np.arange(nmods)+1
    ncwar = ncw.createVariable("ilon", "i", ("lon",))
    ncwar.longname = "indices of longitude points in parent grid"
    ncwar[:] = (np.arange(nx)+i0) * substep
    ncwar = ncw.createVariable("ilat", "i", ("lat",))
    ncwar.longname = "indices of latitude points in parent grid"
    ncwar[:] = indy * substep
    ncwar = ncw.createVariable('lon','f',('lon',))
    ncwar.longname = "longitude (-180 -- 180)"
    ncwar[:] = lon
    ncwar = ncw.createVariable('lat','f',('lat',))
    ncwar.longname = "latitude (90 -- 90)"
    ncwar[:] = lat_s
    ncwar = ncw.createVariable('bathy', "f", ("lat","lon"))
    ncwar.longname = ("Bathymetry")
    ncwar.units = "m"
    ncwar[:] = topo[:]
    ncwar = ncw.createVariable('ratio', "f", ("mode","lat","lon"))
    ncwar.longname = ("Surface/interior modal amplitude ratio")
    ncwar.units = ""
    ncwar = ncw.createVariable('eigenval', "f", ("mode","lat","lon"))
    ncw.longname = 'Modal "reduced" eigenvalue'
    ncw.units = "s/m"
    ncwar = ncw.createVariable('hke', "f", ("mode","lat","lon"))
    ncw.longname = 'z-averaged modal hozirontal kinetic energy'
    ncw.units = "m^2/s^2"
    ncwar = ncw.createVariable('Nref', "f", ("lat","lon"))
    ncw.longname = 'Brunt-vaisala frequency at reference depth'
    ncw.units = "s^{-1}"
    ncw.close()

# store data
def store_val(fwrite,ratio,lam,hke,Nref,jy,indx):
    ncw = Dataset(fwrite, "r+")
    ncw.variables['ratio'][:,jy,indx] = ratio
    ncw.variables['eigenval'][:,jy,indx] = lam
    ncw.variables['hke'][:,jy,indx] = hke
    ncw.variables['Nref'][jy,indx] = Nref
    ncw.close()

#####  ---  Start computing  ---  #####
comm = MPI.COMM_WORLD
ipc = comm.Get_rank()
npcs = comm.Get_size()
if npcs != nproc: raise ValueError('number of proc does not match')

# load data: topo
#N.B.: topo and BVF are assumed to be colocated
nc = Dataset(path_topo, "r")
lon = nc.variables['longitude'][:][::substep]
lat = nc.variables['latitude'][:][::substep]
topo = nc.variables["bathymetry"][:][::substep,::substep]
nc.close()

#topo = np.ma.masked_where(topo<zref, topo)
#
indy, = np.where((lat>lat_min) & (lat<lat_max)) # warning: indy will be overwritten below
indx, = np.where((lon>=lon_min) & (lon<lon_max)) # warning: indy will be overwritten below
j0, i0 = indy.min(), indx.min()

lon, lat, topo = lon[indx], lat[indy], topo[indy,:][:,indx]

# Define chunks in latitude with roughy same number of points to compute
inds = np.where(topo>zref)
jys,invs,jyc = np.unique(inds[0], return_inverse=True, return_counts=True)
npoints = len(inds[0])

indsort = np.argsort(jyc)[::-1] # sorting to have an even repartition of the total number of points
jyc[indsort]
# iproc is the list of nproc lists containing y indices each proc will treat
# ix_prc is the list of lists (1 per array) containing the x-indices to be treated at every y index
iproc = np.reshape(jys[indsort[:len(jyc)-(len(jys)%nproc)]], (-1,nproc)).T.tolist()
for ii in range(len(jys)%nproc):
    iproc[-(ii+1)].append(jys[indsort[-(ii+1)]])
ix_prc = []
for ii in iproc:
    ix_prc.append([inds[1][invs==jj] for jj in ii])
if ipc==0: print("total number of points:", npoints)
nys = len(iproc[ipc])
nxs = sum([len(it) for it in ix_prc[ipc]])
print("proc {0} will do {1} rows, totalling {2} points".format(ipc, nys, nxs))
 
#####  ---  Start Computing  ---  #####
# ISAS: fixed depth
nc = Dataset(path_data+fname, "r")
zbat = nc.variables['depth'][:]
nc.close()

indm = np.array(nmods)-1 # first mode is mode 1 if fs=False

indy = iproc[ipc]
indx = ix_prc[ipc]

lat = lat[indy]

init_netcdf(fwrite.format(ipc), lon, lat, topo[indy,:], indy+j0, i0) 

tmes, tmeb = time.clock(), time.time()
ttot, ttob = time.clock(), time.time()
cnt, cnb = 0, 100
for jj,jy in enumerate(indy):
    nx = len(indx[jj])
    ratio, lam = np.full((nmods,nx),np.nan), np.full((nmods,nx),np.nan)
    hke, Nref = np.full((nmods,nx),np.nan), np.full(nx,np.nan)
    for ii,ix in enumerate(indx[jj]):
        nc = Dataset(path_data+fname, "r")
        nsq = nc.variables['BVF2'][0,:,(j0+jy)*substep,(i0+ix)*substep]
        nc.close()

        nsq[:indzo] = 0
        #if zbat[~nsq.mask][-1] < topo[jy,ix]:
        zz = np.r_[zbat[~nsq.mask],topo[jy,ix]]
        nsq = np.pad(nsq[~nsq.mask].data, (0,1), "edge")
        #else:
            #zz = zbat[~nsq.mask]
            #nsq = nsq[~nsq.mask]

        res = modampratio(nsq, zz, Nzcal, grav, nmods, sm, ksplin, zref)
        # cope with number of eigenvalues found: res is (ratio, lam)
        if res is not None:
            nm = len(res[1])
            ratio[:nm,ii], lam[:nm,ii], hke[:nm,ii], Nref[ii] = res 
    hke /= topo[jy,indx[jj]] # z-averaged
    
    # store result in netCDF file
    store_val(fwrite.format(ipc),ratio,lam,hke,Nref,jj,indx[jj])

    # print timing informations
    cnt += len(indx[jj])
    if cnt >= cnb:  
        print("proc {0}: {1}/{2}".format(ipc, cnt, nxs), 
                ' ({:.1f}, {:.1f}) s'.format(time.clock()-tmes, time.time()-tmeb))
        tmes, tmeb = time.clock(), time.time()
        cnb += 100
print("Proc {0} finished computing {1} points, total time {2:.0f} / {3:.0f} s".format(
        ipc, nxs, time.clock()-ttot, time.time()-ttob))


#####   ---  End of Script  ---  ####


##jy, ix = abs(lat-37).argmin(), abs(lon+32).argmin()
#ix, jy = 294, 335

#sm = 1e-12 # smoothing factor
#ksplin = 1
#zpyc = .8*zref

#nc = Dataset(path_data+fname, "r")
#nsq = nc.variables['BVF2'][0,:,jy,ix]
#nc.close()

#nsq[:indzo] = 0
#nsq[nsq<0] = 0

#zz = np.r_[zbat[~nsq.mask],topo[jy,ix]]
#nsq = np.pad(nsq[~nsq.mask].data, (0,1), "edge")

#prov = modampratio(nsq, zz, Nzcal, sm, ksplin, zref)

#(wo, do), lom, zcheb = SL_chebsolve(nsq[::-1], -zz[::-1], max(modes), Nz=Nzcal, sm=sm, ksplin=ksplin)
#do /= -lom[None,:].real
##(wmod, pmod), ce, (dzr, dzw) = dynmodes(nsq, zz, max(modes), fs=False) 

## compute WKBJ amplitude below zpyc
#indz, = np.where(nsq>0)
#zmax, zmin = -zz[indz.min()], -zz[indz.max()]
#inbz, = np.where((zcheb>zmin) & (zcheb<zmax))
#zchb = zcheb[inbz]
#Nsqin = itp.UnivariateSpline(zz[indz], nsq[indz], k=1, s=sm)(-zchb)
##indz, = np.where((zcheb<=-zpyc) & (Nsqin>0))
#Nsqin[Nsqin<0] = 0
#izref = abs(zchb+zref).argmin()
#amp = np.sqrt(wo[inbz,:]**2*np.sqrt(Nsqin[:,None])          + (do[inbz,:]**2/np.sqrt(Nsqin[:,None]))).mean(axis=0)

#print("amp is",amp)

## normalize such that do is 1 (in ampliute) at zref
#norm = amp * Nsqin[izref]**(1/4)
#do /= norm[None,:]
#wo /= norm[None,:]
#amp /= norm

#print("amp is",amp[0]) # normalized amp is the same for every modes

#ratio = do[-1,:]**2

#print(ratio)
#print(prov[0])
#print(lom, prov[1])


## In[223]:



## In[208]:


#imod = 2
#ampd = amp[imod]*Nsqin**(1/4)
#ampw = amp[imod]/Nsqin**(1/4)

#fig, axs = plt.subplots(1, 3, sharey=True)
#ax = axs[0]
#ax.plot(do[:,imod], zcheb)
#ax.plot(np.c_[-ampd, ampd], zchb, "--", color="grey")

#ax = axs[1]
#ax.plot(wo[:,imod], zcheb)
#ax.plot(np.c_[-ampw, ampw], zchb, "--", color="grey")
#ax.set_xlim(np.array([-1,1])*1.2*abs(wo[:,imod]).max())

#ax = axs[2]
#ax.plot(np.sqrt(wo[inbz,imod]**2*np.sqrt(Nsqin)          + do[inbz,imod]**2/np.sqrt(Nsqin)), zchb, "+-")
#ab = ax.twiny()
#ab.plot(Nsqin, zchb, ".-k")

#for ax in axs:
    #ax.grid(True)


## In[226]:


## compute over a sub-domain

#nc = Dataset(path_data+fname, "r")
#zbat = nc.variables['depth'][:]
#nc.close()


## In[234]:


#plt.figure()
#plt.pcolormesh(lon[indx], lat[indy], np.log10(ratio[-1,:,:]), cmap="RdBu_r", vmin=-1.5, vmax=1.5)
#plt.colorbar(fraction=.02, extend="both")
#plt.contour(lon[indx], lat[indy], topo[indy,:][:,indx], levels=[1000, 2000, 3500], colors="grey", linewidths=.8)
#plt.grid(True)
##plt.gca().set_aspect(1/np.sin(np.deg2rad(lat[indy].mean())))

#jj, ii = np.unravel_index(ratio[4,:,:].argmin(), (ny,nx))
#print(ii+i0, jj+j0, lon[indx][ii], lat[indy][jj])


## In[324]:


#len(inds[0])


## In[307]:


#import gsw
#gsw.distance(lon[:2,None],lat[])


## In[326]:


## make chunks (y)
#lat_min = -80
#lat_max = 80
#nproc = 6

   


## In[327]:



## In[312]:


#iproc[0][50]
#(ix_prc[0][50]==inds[1][inds[0]==iproc[0][50]]).all()


## In[55]:


#sm = 1e-12
#ncal = itp.UnivariateSpline(zz, nsq, k=1, s=sm)(-zcheb)

#fig, axs = plt.subplots(1, 3, sharey=True, figsize=(8,4))
#ax = axs[0]
#ax.plot(nsq, -zz, '-k')
#ax.plot(ncal, zcheb, ".-", color="tab:blue")
#ax.grid(True)
#ax.ticklabel_format(style='sci',scilimits=(-2,4))

#axs[1].plot(wb, zcheb, color="tab:red")
#axs[2].plot(ub, zcheb, color="tab:blue")

#plt.ylim([-zref, 0])
#for ax in axs:
    #ax.grid(True)

