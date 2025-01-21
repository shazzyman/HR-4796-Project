#!/usr/bin/env python

# Imports
import sys
import os
import emcee
import h5py
import numpy as np
import numba as nb
from scipy import interpolate
from scipy import optimize
from schwimmbad import MPIPool
from mpi4py import MPI

# User defined functions
from getstellarmodel import getstellarmodel_fun
from disk_model import disk_model_fun
import shared_constants as co
import disk_parameters as dpar


os.environ["OMP_NUM_THREADS"] = "1"
# sys.path.append('/home/jarnold/DiskModelShare') # *Need this to run on Memex, put the directory where the code resides.*


# Calculate the phase angle of each portion of the disk
@nb.jit(nb.types.Tuple((nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :]))
        (nb.float64, nb.int64, nb.int64), nopython=True)
def calc_phase_ang(pixsc, npix, zpix):
    # Create 3d arrays with particle number density and phase angles
    sigma_up = np.zeros((npix, npix, zpix))  # Number density for each pixel
    sigma_y = np.zeros((npix, npix, zpix))   # Where each volumetric pixel ends up after projection
    sigma_d = np.zeros((npix, npix, zpix))   # Distance from each voxel to star in meters
    phang = np.zeros((npix, npix, zpix))     # Phase angle for each volume pixel

    hz_center_rmax = pixsc * dpar.rmax * np.tan(np.pi * (1. - dpar.ophi / 180.) / 2)    # Height of disk at breakpoint
    sigma_up_rmax_xsigma = ((dpar.rmax ) ** dpar.xsigma) / (2 * hz_center_rmax)         # Surface density of disk at breakpoint
    sigma_up_rmax_xsigmahalo = ((dpar.rmax) ** dpar.xsigmahalo) / (2 * hz_center_rmax)  # Surface density of the halo
    sigma_up_halo_norm = sigma_up_rmax_xsigma/sigma_up_rmax_xsigmahalo
    for xi in range(0, npix):
        for yi in range(0, npix):
            dxy_center = np.sqrt((xi - npix / 2.) ** 2 + (yi - npix / 2.) ** 2)     # Horizontal distance to star
            hz_center = dxy_center * np.tan(np.pi * (1. - dpar.ophi / 180.) / 2)    # Total height of the disk at eaxh x, y point
            for zi in range(0, zpix):
                dcenter = np.sqrt((xi - npix / 2.) ** 2 + (yi - npix / 2.) ** 2 + (zi - zpix / 2.) ** 2)    # Total distance from the voxel to the star in pixels
                d_y0 = np.sqrt((xi - npix / 2.) ** 2 + yi ** 2.)    # Distance from bottom of x-y projection
                sigma_d[xi, yi, zi] = (1. / pixsc) * dcenter * co.rAU   # Total distance from the voxel to the star in meters
                #  Number density and phase angles for disk
                if ((dxy_center >= pixsc * dpar.rmin) and (dxy_center <= pixsc * dpar.rmax) and
                        (zi >= (zpix / 2. - hz_center)) and (zi <= (zpix / 2. + hz_center))):
                    sigma_y[xi, yi, zi] = (yi - npix / 2.) * dpar.cos_i - (zi - zpix / 2.) * dpar.sin_i + npix / 2.
                    sigma_up[xi, yi, zi] = ((dxy_center / pixsc) ** dpar.xsigma) / (2 * hz_center)
                    tht = np.arctan((zi - zpix / 2) / dxy_center)
                    phi = np.arccos(((npix / 2.) ** 2 + dxy_center ** 2 - d_y0 ** 2) / (2 * (npix / 2.) * dxy_center))
                    phang[xi, yi, zi] = np.arccos(dpar.cos_i * np.sin(-tht) + dpar.sin_i * np.cos(-tht) * np.cos(phi))
                #  Number density and phase angles for halo
                if ((dxy_center > pixsc * dpar.rmax) and (dxy_center <= pixsc * dpar.rmaxhalo) and
                        (zi >= (zpix / 2. - hz_center)) and (zi <= (zpix / 2. + hz_center))):
                    sigma_y[xi, yi, zi] = (yi - npix / 2.) * dpar.cos_i - (zi - zpix / 2.) * dpar.sin_i + npix / 2.
                    sigma_up[xi, yi, zi] = sigma_up_halo_norm*((dxy_center / pixsc) ** dpar.xsigmahalo) / (2 * hz_center)
                    tht = np.arctan((zi - zpix / 2) / dxy_center)
                    phi = np.arccos(((npix / 2.) ** 2 + dxy_center ** 2 - d_y0 ** 2) / (2 * (npix / 2.) * dxy_center))
                    phang[xi, yi, zi] = np.arccos(dpar.cos_i * np.sin(-tht) + dpar.sin_i * np.cos(-tht) * np.cos(phi))

    return sigma_up, sigma_y, sigma_d, phang


# Uniform prior for each parameter
#   Minimum grain size between 0.01 and 1 micron
#   Power law grain size distribution exponent between -5 and -1
#   Total disk mass order of magnitude between 22 and 26 (kg)
#   Volume fraction of each compositional component total to 1
def lnprior(param_list):
    amin, aexp, mexp, f_asi, f_ac, f_wi, f_th, f_tr, f_fe, lnf = param_list
    if 0.01 < amin < 1. and -5. < aexp < -1. and 22. < mexp < 26. and 0. < f_asi < 1. and 0. < f_ac < 1. \
            and 0. < f_wi < 1. and 0. < f_th < 1. and 0. < f_tr < 1. and 0. < f_fe < 1. \
            and 0.99 < f_asi + f_ac + f_wi + f_th + f_tr + f_fe < 1.01 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

# Evaluate the Chi-square likelihood
def lnlike(param_list):
    amin, aexp, mexp, f_asi, f_ac, f_wi, f_th, f_tr, f_fe, lnf = param_list
    ld, ldp, lx, ly, fluxmodel = disk_model_fun(wl, amin, amax, aexp, mexp, f_asi, f_ac, f_wi,
                                                    f_th, f_tr, f_fe, m_asi, m_ac, m_wi, m_th, m_tr,
                                                    m_fe, pixsc, npix, zpix, sigma_up, sigma_y, sigma_d,
                                                    phang, scatt_type, projected_separation_pts_au)
    inv_sigma2 = 1.0 / (fluxerr ** 2 + fluxmodel ** 2 * np.exp(2 * lnf))
    # print(inv_sigma2)
    # print(fluxerr)
    # print(fluxmodel)
    # print(lnf)
    return -0.5*(np.sum((fluxmeasured-fluxmodel)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(param_list):
    lp = lnprior(param_list)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(param_list)


# Bruggeman mixing rule for porous spheres
def bruggeman2(eff, *args):
    effc = eff[0:eff.size//2] + 1j*eff[eff.size//2:]
    e1=args[0]+1j*args[1]
    e2=(1.0 + np.zeros(e1.size)) + 1j*(0.0 + np.zeros(e1.size))
    minc = 0.25 * (e1 - effc) / (e1 + 2 * effc) + 0.75 * (e2 - effc) / (e2 + 2 * effc)
    return np.concatenate((np.real(minc), np.imag(minc)))


# The emcee docs recommend making arguments passed to the model from the likelihood function
    # global to increase efficiency when using the multiprocessing package and Pool for the
    # parallelization. I switched to using mpi but have left this implementation in case I
    # or someone want to switch back to Pool.
global wl, amax, m_asi, m_ac, m_wi, m_th, m_tr, m_fe, projected_separation_pts_au, fluxmeasured, fluxerr


# Set wavelength range and maximum grain size   *Edit for specific disk*
wlo = 0.5385    # Minimum wavelength
whi = 0.9877    # Maximum wavelength
nwl = 10        # Number of wavelengths
wl = np.linspace(wlo, whi, num=nwl)
amax = 100.


# The stellar model is not needed to calculate the flux ratio, but I have this in here in case some.
calc_stellarmodel = True
if calc_stellarmodel:
    # Stellar parameters    *Edit for specific disk*
    Tstar = 3720      # [K] Stellar temperature
    lstar = 0.11      # [Lsun] Stellar luminosity
    Mstar = 0.6       # [Msun] Stellar mass

    # Get stellar model
    Rstar = np.sqrt(lstar*co.lsun/(4.*np.pi*co.sig*Tstar**4))/co.Rsun
    logg = np.log10(Mstar)-2*np.log10(Rstar) + 4.437
    starmodelfull, wlfull = getstellarmodel_fun(Tstar, logg=logg,
                                                modeldir='./stellarmodels/')  # Stellar irradiance [Jy/ster]
    starmodel = np.interp(wl, wlfull, starmodelfull)
    Lstar = 4*np.pi*(Rstar**2.)*starmodel


# Load dust refractive indices
porosity = False # True if using porous spheres
scatt_type = 'Agglomerate'   # *Choices are 'Mie', 'Porous', 'Agglomerate'*
n0 = 1.0 + np.zeros(wl.size)
k0 = 0.1 + np.zeros(wl.size)
m_guess = np.concatenate((n0, k0))   # Used if 'Porous' is chosen
m_brug = np.zeros(m_guess.shape)

# Astrosilicate
nk_file_asi = './optical_constants/Astro_silicate_Draine_2003.txt'
nkdat_asi = np.loadtxt(nk_file_asi, skiprows=5, unpack=True)
wlnk_asi = np.flipud(nkdat_asi[0, :])
nf_asi = np.flipud(1 + nkdat_asi[3, :])
kf_asi = np.flipud(nkdat_asi[4, :])
n_asi = np.interp(wl, wlnk_asi, nf_asi)
k_asi = np.interp(wl, wlnk_asi, kf_asi)
if porosity:
    m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_asi,k_asi))
    m_asi = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
else:
    m_asi = n_asi - 1j*k_asi

# Amorphous carbon
nk_file_ac = './optical_constants/amorphousC_ACAR_Zubko.txt'
wlnk_ac, nf_ac, kf_ac = np.loadtxt(nk_file_ac, skiprows=2, usecols=(0, 1, 2), unpack=True)
n_ac = np.interp(wl, wlnk_ac, nf_ac)
k_ac = np.interp(wl, wlnk_ac, kf_ac)
if porosity:
    m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_ac,k_ac))
    m_ac = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
else:
    m_ac = n_ac - 1j*k_ac

# Water Ice
nk_file_wi = './optical_constants/waterice_Henning.txt'
wlnk_wi, nf_wi, kf_wi = np.loadtxt(nk_file_wi, skiprows=2, usecols=(0, 1, 2), unpack=True)
n_wi = np.interp(wl, wlnk_wi, nf_wi)
k_wi = np.interp(wl, wlnk_wi, kf_wi)
if porosity:
    m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_wi,k_wi))
    m_wi = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
else:
    m_wi = n_wi - 1j*k_wi

# Tholins
nk_file_th = './optical_constants/tholins.txt'
wlnk_th, nf_th, kf_th = np.loadtxt(nk_file_th, skiprows=2, usecols=(0, 1, 2), unpack=True)
n_th = np.interp(wl, wlnk_th, nf_th)
k_th = np.interp(wl, wlnk_th, kf_th)
if porosity:
    m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_th,k_th))
    m_th = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
else:
    m_th = n_th - 1j*k_th

# Troilite
nk_file_tr = './optical_constants/troilite.txt'
wlnk_tr, nf_tr, kf_tr = np.loadtxt(nk_file_tr, skiprows=2, usecols=(0, 1, 2), unpack=True)
n_tr = np.interp(wl, wlnk_tr, nf_tr)
k_tr = np.interp(wl, wlnk_tr, kf_tr)
if porosity:
    m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_tr,k_tr))
    m_tr = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
else:
    m_tr = n_tr - 1j*k_tr

# 0Fe
nk_file_fe = './optical_constants/iron_Henning.txt'
wlnk_fe, nf_fe, kf_fe = np.loadtxt(nk_file_fe, skiprows=2, usecols=(0, 1, 2), unpack=True)
n_fe = np.interp(wl, wlnk_fe, nf_fe)
k_fe = np.interp(wl, wlnk_fe, kf_fe)
if porosity:
    m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_fe,k_fe))
    m_fe = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
else:
    m_fe = n_fe - 1j*k_fe


# *Import disk data*
disk_dir = './AUMic'
wl10nw, AU10NW, ERR10NW = np.loadtxt(disk_dir+'/NW10AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl15nw, AU15NW, ERR15NW = np.loadtxt(disk_dir+'/NW15AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl20nw, AU20NW, ERR20NW = np.loadtxt(disk_dir+'/NW20AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl25nw, AU25NW, ERR25NW = np.loadtxt(disk_dir+'/NW25AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl30nw, AU30NW, ERR30NW = np.loadtxt(disk_dir+'/NW30AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl35nw, AU35NW, ERR35NW = np.loadtxt(disk_dir+'/NW35AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl40nw, AU40NW, ERR40NW = np.loadtxt(disk_dir+'/NW40AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl45nw, AU45NW, ERR45NW = np.loadtxt(disk_dir+'/NW45AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl10se, AU10SE, ERR10SE = np.loadtxt(disk_dir+'/SE10AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl15se, AU15SE, ERR15SE = np.loadtxt(disk_dir+'/SE15AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl20se, AU20SE, ERR20SE = np.loadtxt(disk_dir+'/SE20AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl25se, AU25SE, ERR25SE = np.loadtxt(disk_dir+'/SE25AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl30se, AU30SE, ERR30SE = np.loadtxt(disk_dir+'/SE30AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl35se, AU35SE, ERR35SE = np.loadtxt(disk_dir+'/SE35AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl40se, AU40SE, ERR40SE = np.loadtxt(disk_dir+'/SE40AUSpec.txt', skiprows=2, max_rows=49, unpack=True)
wl45se, AU45SE, ERR45SE = np.loadtxt(disk_dir+'/SE45AUSpec.txt', skiprows=2, max_rows=49, unpack=True)

# Average NW and SE sides of disk
wl10 = wl10nw

AU10full = (AU10NW[:-2] + AU10SE[:-2])/2.
AU20full = (AU20NW[:-2] + AU20SE[:-2])/2.
AU30full = (AU30NW[:-2] + AU30SE[:-2])/2.
AU40full = (AU40NW[:-2] + AU40SE[:-2])/2.

ERR10full = np.sqrt(ERR10NW[:-2]**2. + ERR10SE[:-2]**2.)/2.
ERR20full = np.sqrt(ERR20NW[:-2]**2. + ERR20SE[:-2]**2.)/2.
ERR30full = np.sqrt(ERR30NW[:-2]**2. + ERR30SE[:-2]**2.)/2.
ERR40full = np.sqrt(ERR40NW[:-2]**2. + ERR40SE[:-2]**2.)/2.

# Interpolate data to chosen wavelengths/resolution
y_spline = interpolate.splrep(0.0001*wl10[:-2], AU10full, k=3, t=None)
dy_spline = interpolate.splrep(0.0001*wl10[:-2], ERR10full, k=3, t=None)
AU10 = interpolate.splev(wl, y_spline, der=0)
ERR10 = interpolate.splev(wl, dy_spline, der=0)
y_spline = interpolate.splrep(0.0001*wl10[:-2], AU20full, k=3, t=None)
dy_spline = interpolate.splrep(0.0001*wl10[:-2], ERR20full, k=3, t=None)
AU20 = interpolate.splev(wl, y_spline, der=0)
ERR20 = interpolate.splev(wl, dy_spline, der=0)
y_spline = interpolate.splrep(0.0001*wl10[:-2], AU30full, k=3, t=None)
dy_spline = interpolate.splrep(0.0001*wl10[:-2], ERR30full, k=3, t=None)
AU30 = interpolate.splev(wl, y_spline, der=0)
ERR30 = interpolate.splev(wl, dy_spline, der=0)
y_spline = interpolate.splrep(0.0001*wl10[:-2], AU40full, k=3, t=None)
dy_spline = interpolate.splrep(0.0001*wl10[:-2], ERR40full, k=3, t=None)
AU40 = interpolate.splev(wl, y_spline, der=0)
ERR40 = interpolate.splev(wl, dy_spline, der=0)

# Put fluxes and error each into one array that MCMC can fit
fluxmeasured = np.append(np.append(AU10, AU20), np.append(AU30, AU40))
fluxerr = np.append(np.append(ERR10, ERR20), np.append(ERR30, ERR40))
projected_separation_pts_au = np.array([10., 20., 30., 40.]) # *Choose projected separations to fit (in AU)*

# Create 3D array with particle surface density
pixsc = 1. / dpar.pxs  # *pixels per AU*
npix = np.int(np.round(pixsc * 170))    # *Choose number of pixels in model (in plane of disk)*
zpix = np.int(np.round(pixsc * 10))     # *Choose number of pixels in model (vertical)*
sigma_up, sigma_y, sigma_d, phang = calc_phase_ang(pixsc, npix, zpix)


# Set up MCMC
coords = np.random.randn(8, 10) # This is (number of chains, number of variables)
nwalkers, ndim = coords.shape
max_n = 2 # *max iterations* Should be much larger ~1k-10k the low number is just to test.

# Initialize chain with uniform distribution
pos_min = np.array([0.01, -5., 22., 0., 0., 0., 0., 0., 0., -10.])  # *Posterior min and max*
pos_max = np.array([1., -1., 26., 1., 1., 1., 1., 1., 1., 1.])
pos = [[] for i in range(nwalkers)]
psize = pos_max - pos_min
np.random.seed(42)
for i in range(nwalkers):
    amin_i = pos_min[0] + psize[0] * np.random.rand()
    aexp_i = pos_min[1] + psize[1] * np.random.rand()
    mexp_i = pos_min[2] + psize[2] * np.random.rand()
    f = np.random.dirichlet([10, 2, 2, 2, 1, 1])    # I don't know if this is the best way, but this gives a reasonable
                                                        # array of walkers for the volume fraction of each component
                                                        # that sums to 1
    lnf_i = pos_min[9] + psize[9] * np.random.rand()
    pos[i] = np.array([amin_i, aexp_i, mexp_i, f[0], f[1], f[2], f[3], f[4], f[5], lnf_i])
pos = np.array(pos)
filename = "emcee_Agl_6C_100X10X10.h5"   # *File name to save posteriors* Can be used to continue calculation or make corner plots
backend = emcee.backends.HDFBackend(filename)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    backend.reset(nwalkers, ndim)  # Comment out this line to continue previous calculation working from a
                                   # previously saved back end. Otherwise, reset the backend for a new run.
                                   # The rank == 0 makes sure that the reset only occurs once when using mpi.

#Do MCMC the pool will have however many processors you specified in the command line
# !mpiexec -n {nprocessors} python disk_model_mcmc_wrapper.py
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
    pos, prob, state = sampler.run_mcmc(pos, max_n, progress=True, skip_initial_state_check=True)


# Save input parameters to plot modeled spectra later
initial = h5py.File('MCMC_Agl_6C_100X10X10.hdf5', 'w')
initial.create_dataset('wl', data=wl)
initial.create_dataset('AU10', data=AU10)
initial.create_dataset('ERR10', data=ERR10)
initial.create_dataset('AU20', data=AU20)
initial.create_dataset('ERR20', data=ERR20)
initial.create_dataset('AU30', data=AU30)
initial.create_dataset('ERR30', data=ERR30)
initial.create_dataset('AU40', data=AU40)
initial.create_dataset('ERR40', data=ERR40)
initial.create_dataset('amax', data=amax)
initial.create_dataset('m_asi', data=m_asi)
initial.create_dataset('m_ac', data=m_ac)
initial.create_dataset('m_wi', data=m_wi)
initial.create_dataset('m_th', data=m_th)
initial.create_dataset('m_tr', data=m_tr)
initial.create_dataset('m_fe', data=m_fe)
initial.create_dataset('pixsc', data=pixsc)
initial.create_dataset('npix', data=npix)
initial.create_dataset('zpix', data=zpix)
initial.create_dataset('sigma_up', data=sigma_up)
initial.create_dataset('sigma_y', data=sigma_y)
initial.create_dataset('sigma_d', data=sigma_d)
initial.create_dataset('phang', data=phang)
initial.create_dataset('projected_separation_pts_au', data=phang)
initial.close()