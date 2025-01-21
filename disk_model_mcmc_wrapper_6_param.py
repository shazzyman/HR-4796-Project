#!/usr/bin/env python
"""
This Python script is designed to perform an MCMC (Markov Chain Monte Carlo) analysis of a 6 paramter dust disk model, leveraging MPI parallelization for 
efficient computation on Linux systems. It uses various libraries like emcee for MCMC sampling, mpi4py for distributed computing, and numba to optimize
numerical calculations. The script calculates the phase angles, particle distributions, and fluxes for a dust disk, given physical parameters and observational 
data. It includes custom definitions for dust refractive indices and stellar modeling, allowing precise computation of scattering properties and phase functions.

The script reads precomputed scattering data from HDF5 files and uses the parameters to simulate the disk's luminosity and flux based on user-defined scattering 
types (Mie, Porous, or Agglomerate). It initializes MCMC walkers to explore the parameter space, saving posterior distributions to HDF5 files for further analysis. 
MPI parallelization, enabled with the schwimmbad package, allows the script to scale across multiple processors, making it suitable for high-performance Linux clusters. 
The resulting output includes fitted parameters and data files, which can be used for generating synthetic spectra and studying debris disks in astrophysics.
"""
# Imports
import sys
import os
import emcee
import h5py
import numpy as np
import numba as nb
from scipy import interpolate
from scipy.ndimage.interpolation import rotate
from scipy import optimize
from schwimmbad import MPIPool
from mpi4py import MPI

# User defined functions
from getstellarmodel import getstellarmodel_fun
from disk_model_new import disk_model_fun_new
import shared_constants as co
import disk_parameters as dpar
import square_apertures as sq_ap
from square_apertures import square_aperture_calcs

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
#   Minimum grain size between 0.01 and 4 micron
#   Power law grain size distribution exponent between -5 and -1
#   Total disk mass order of magnitude between 27 and 33 (kg)
#   Volume fraction of each compositional component total to 1
def lnprior_new(param_list):
    amin, mexp, f_asi, f_ac, f_wi, lnf = param_list
    if 2. < amin < 10. and 30. < mexp < 34 and 0. <= f_asi <= 1. and 0. <= f_ac <= 1. \
            and 0. <= f_wi <= 1. \
            and 0.99 < f_asi + f_ac + f_wi < 1.01 and -1.0 < lnf < 1.0:
        return 0.0
    return -np.inf

# Evaluate the Chi-square likelihood
def lnlike_new(param_list):
    amin, mexp, f_asi, f_ac, f_wi, lnf = param_list
    ld, ldp, lx, ly, fluxmodel_trash = disk_model_fun_new(wl, amin, amax, aexp, mexp, f_asi, f_ac, f_wi,
                                                    m_asi, m_ac, m_wi, pixsc, npix, zpix, sigma_up, sigma_y, sigma_d,
                                                    phang, scatt_type, projected_separation_pts_au)
    
    ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
    ldp_rotated[np.where(ldp_rotated< 0)] = 0
    
    fluxmodel = np.zeros(nwl)
    for i in range(0, nwl):
        aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y) #calculating medians, these apertures matched the ones in Rodigas Paper
        fluxmodel[i] = medians #inputting medians into array that can be used as fluxmodel
        
        
    inv_sigma2 = 1.0 / (fluxerr ** 2 + fluxmodel ** 2 * np.exp(2 * lnf)) 
    # print('the value of inv sig is', inv_sigma2)
    # print('fluxerr is', fluxerr)
    # print('fluxmodel is', fluxmodel)
    # print('lnf is', lnf)
    #print('What we'r're looking for:', np.sum((fluxmeasured-fluxmodel)**2))
    return -0.5*(np.sum((fluxmeasured-fluxmodel)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob_new(param_list):
    lp = lnprior_new(param_list)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_new(param_list)


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
global wl, amax, m_asi, m_ac, m_wi, projected_separation_pts_au, fluxmeasured, fluxerr, aexp

# Set wavelength range and maximum grain size   *Edit for specific disk*
wlo = 0.5737    # Minimum wavelength
whi = 3.7565    # Maximum wavelength
nwl = 11       # Number of wavelengths
wl = np.linspace(wlo, whi, num=nwl)
amax = 100.
aexp = -3.0

# The stellar model is not needed to calculate the flux ratio, but I have this in here in case some.
calc_stellarmodel = True
if calc_stellarmodel:
    # Stellar parameters    *Edit for specific disk*
    Tstar = 9250      # [K] Stellar temperature
    lstar = 21.1      # [Lsun] Stellar luminosity
    Mstar = 2.5       # [Msun] Stellar mass

    # Get stellar model
    Rstar = np.sqrt(lstar*co.lsun/(4.*np.pi*co.sig*Tstar**4))/co.Rsun
    logg = np.log10(Mstar)-2*np.log10(Rstar) + 4.437
    starmodelfull, wlfull = getstellarmodel_fun(Tstar, logg=logg,
                                                modeldir='./stellarmodels/')  # Stellar irradiance [Jy/ster]
    starmodel = np.interp(wl, wlfull, starmodelfull)
    Lstar = 4*np.pi*(Rstar**2.)*starmodel
    
    #Load dust refractive indices
    porosity = False #True if using porous spheres
    scatt_type = dpar.scatt_type 
    n0 = 1.0 +np.zeros(wl.size)
    k0 = 0.1 + np.zeros(wl.size)
    m_guess = np.concatenate((n0, k0)) #Used if 'Porous' chosen
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
    
       
# #Astro/wi
# nk_file_asi = './optical_constants/brug_astrosil80_waterice_por30.txt'
# nkdat_asi = np.loadtxt(nk_file_asi, unpack=True)
# wlnk_asi = np.flipud(nkdat_asi[0, :])
# nf_asi = np.flipud(1 + nkdat_asi[1, :])
# kf_asi = np.flipud(nkdat_asi[2, :])
# n_asi = np.interp(wl, wlnk_asi, nf_asi)
# k_asi = np.interp(wl, wlnk_asi, kf_asi)
# if porosity:
#     m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_asi,k_asi))
#     m_asi = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
# else:
#     m_asi = n_asi - 1j*k_asi
    
    
# #Orthopyro
# nk_file_asi = './optical_constants/brug_olivineIP80_orthopyr_por0.txt'
# nkdat_asi = np.loadtxt(nk_file_asi, unpack=True)
# wlnk_asi = np.flipud(nkdat_asi[0, :])
# nf_asi = np.flipud(1 + nkdat_asi[1, :])
# kf_asi = np.flipud(nkdat_asi[2, :])
# n_asi = np.interp(wl, wlnk_asi, nf_asi)
# k_asi = np.interp(wl, wlnk_asi, kf_asi)
# if porosity:
#     m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_asi,k_asi))
#     m_asi = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
# else:
#     m_asi = n_asi - 1j*k_asi
    

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

# *Import disk data*
disk_dir = './DiskModelShare'
wl, fl, err = np.loadtxt('Rodigas_export_new.csv', skiprows = 1, unpack=True, delimiter = ',', usecols=(1,2,6))
# Put fluxes and error each into one array that MCMC can fit
fluxmeasured = fl
fluxerr = err
projected_separation_pts_au = np.array([75.]) # *Choose projected separations to fit (in AU)*

# Create 3D array with particle surface density
pixsc = 1. / dpar.pxs  # *pixels per AU*
npix = int(np.round(pixsc * 350))    # *Choose number of pixels in model (in plane of disk)*
zpix = int(np.round(pixsc * 10))     # *Choose number of pixels in model (vertical)*
sigma_up, sigma_y, sigma_d, phang = calc_phase_ang(pixsc, npix, zpix)

# Set up MCMC
coords = np.random.randn(60, 6) # This is (number of chains, number of variables)
nwalkers, ndim = coords.shape
max_n = 10000  # *max iterations* Should be much larger ~1k-10k the low number is just to test.

# Initialize chain with uniform distribution
pos_min = np.array([2, 30., 0.6, 0., 0., -0.1])  # *Posterior min and max*
pos_max = np.array([10., 34, 1., 1., 1., 0.1])
pos = [[] for i in range(nwalkers)]
psize = pos_max - pos_min
np.random.seed(42)
for i in range(nwalkers):
    amin_i = pos_min[0] + psize[0] * np.random.rand()
    mexp_i = pos_min[1] + psize[1] * np.random.rand()
    mfsi_i = pos_min[2] + psize[2] * np.random.rand()
    f_other = 1 - mfsi_i
    f_ac = 0.5*f_other
    f_wi = 0.5*f_other
    # f = np.random.dirichlet([10, 2, 2,])    # I don't know if this is the best way, but this gives a reasonable
    #                                                     # array of walkers for the volume fraction of each component
    #                                                     # that sums to 1
    lnf_i = pos_min[5] + psize[5] * np.random.rand()
    pos[i] = np.array([amin_i, mexp_i, mfsi_i, f_ac, f_wi, lnf_i])
pos = np.array(pos)
filename = "shaz_emcee_Agl_6C_60X6X500_Astro_silicate.h5"   # *File name to save posteriors* Can be used to continue calculation or make corner plots
backend = emcee.backends.HDFBackend(filename)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# if rank == 0:
#     backend.reset(nwalkers, ndim)  # Comment out this line to continue previous calculation working from a
                                    # previously saved back end. Otherwise, reset the backend for a new run.
                                    # The rank == 0 makes sure that the reset only occurs once when using mpi.

#Do MCMC the pool will have however many processors you specified in the command line
# !mpiexec -n {nprocessors} python disk_model_mcmc_wrapper.py
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_new, pool=pool, backend=backend)
    pos, prob, state = sampler.run_mcmc(pos, max_n, store = True, progress=True, skip_initial_state_check=True)

pos = np.array(pos)
backend = emcee.backends.HDFBackend('shaz_emcee_Agl_6C_60X6X500_Astro_silicate.h5')

# Save input parameters to plot modeled spectra later
initial = h5py.File('shaz_MCMC_Agl_6C_60X6X500_Astro_silicate.hdf5', 'w')
initial.create_dataset('wl', data=wl)
initial.create_dataset('amax', data=amax)
initial.create_dataset('m_asi', data=m_asi)
initial.create_dataset('m_ac', data=m_ac)
initial.create_dataset('m_wi', data=m_wi)
initial.create_dataset('pixsc', data=pixsc)
initial.create_dataset('npix', data=npix)
initial.create_dataset('zpix', data=zpix)
initial.create_dataset('sigma_up', data=sigma_up)
initial.create_dataset('sigma_y', data=sigma_y)
initial.create_dataset('sigma_d', data=sigma_d)
initial.create_dataset('phang', data=phang)
initial.create_dataset('projected_separation_pts_au', data=phang)
initial.close()


