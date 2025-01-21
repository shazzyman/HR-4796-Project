#!/usr/bin/env python

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
from dictionary import chem_dict
from disk_parameters import comptypes_array


# User defined functions
from getstellarmodel import getstellarmodel_fun
from disk_model_func import param_list_final
from disk_model_func import disk_model_fun
import shared_constants as co
import disk_parameters as dpar
import square_apertures as sq_ap
from square_apertures import square_aperture_calcs

os.environ["OMP_NUM_THREADS"] = "1"
# sys.path.append('/home/jarnold/DiskModelShare') # *Need this to run on Memex, put the directory where the code resides.*
# if dpar.amin == False:
#     amin = 3.0
    
# if dpar.aexp == False:
#     aexp = -3.
    
# if dpar.mexp == False:
#     mexp = 3.
    
# if dpar.mll == False:
#     mll = 0.
    
# Calculate the phase angle of each portion of the disk
@nb.jit(nb.types.Tuple((nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :]))
        (nb.float64, nb.int64, nb.int64), nopython=True)

def calc_phase_ang_surf_dens(pixsc, npix, zpix):
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
def lnprior(param_list):
    
        
    comptypes_array = dpar.comptypes_array   
    
    amin, mexp, f_1, f_2, f_3, mll = param_list
    
    f_array = []
    f_array = np.append(f_array, [f_1, f_2, f_3])
    # for i in range(0, len(comptypes_array)-1):
    #     0 <= f_array[i] <= 1 
    for i in range(0, len(f_array)):
        if i > (len(comptypes_array) -1):

            f_array[i] = 0
            
            
    if 2. < amin < 10. and -5. < aexp < -1 and 30. < mexp < 34 and 0. <= f_1 <= 1. and 0. <= f_2 <= 1. \
            and 0. <= f_3 <= 1. \
            and 0.99 < f_1 + f_2 + f_3 < 1.01 and -1.0 < mll < 1.0:
        return 0.0
    return -np.inf


# Evaluate the Chi-square likelihood
def lnlike(param_list):
    comptypes_array = dpar.comptypes_array
    amin, mexp, f_1, f_2, f_3, mll = param_list
    ld, ldp, lx, ly, fluxmodel_trash = disk_model_fun(wl, amin, amax, aexp, mexp, f_1, f_2, f_3, m_1, m_2, m_3, 
                       pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, scatt_type, projected_separation_pts_au)
        
    ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
    ldp_rotated[np.where(ldp_rotated< 0)] = 0
    
        
    fluxmodel = np.zeros(nwl)
    for i in range(0, nwl):
            aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y) #calculating medians, these apertures matched the ones in Rodigas Paper
            fluxmodel[i] = medians #inputting medians into array that can be used as fluxmodel
        
        
    inv_sigma2 = 1.0 / (fluxerr ** 2 + fluxmodel ** 2 * np.exp(2 * mll)) 
       
    return -0.5*(np.sum((fluxmeasured-fluxmodel)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(param_list):
    comptypes_array= dpar.comptypes_array
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
global wl, amax, projected_separation_pts_au, fluxmeasured, fluxerr, ndim
if dpar.amin == False:
    amin = 3.0
    
if dpar.aexp == False:
    aexp = -3.
    
if dpar.mexp == False:
    mexp = 3.
    

if dpar.mll == False:
    mll = 0.
    
m_1 = chem_dict[comptypes_array[0]][1]
m_2 = chem_dict[comptypes_array[1]][1]
m_3 = chem_dict[comptypes_array[2]][1]
 

# Set wavelength range and maximum grain size   *Edit for specific disk*
wlo = 0.5737    # Minimum wavelength
whi = 3.7565    # Maximum wavelength
nwl = 11       # Number of wavelengths
wl = np.linspace(wlo, whi, num=nwl)
amax = 100.

#Complex index of refraction retrieval, remember to un-comment out what yu need!
m_dict = []

for comptype in comptypes_array:
    m_dict = np.append(m_dict, [m_1, m_2, m_3])
    


    
    
# m_4 = chem_dict[dpar.comptypes[3]][1]
# m_5 = chem_dict[dpar.comptypes[4]][1]
# m_6 = chem_dict[dpar.comptypes[5]][1]


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
sigma_up, sigma_y, sigma_d, phang = calc_phase_ang_surf_dens(pixsc, npix, zpix)

# Set up MCMC

pos_min = np.zeros(10)
pos_max = np.zeros(10)


fin_pos_min = np.delete(pos_min, range(0,10))

fin_pos_max = np.delete(pos_max, range(0,10))
                    
coords = np.random.randn(60, len(param_list_final)) # This is (number of chains, number of variables)
nwalkers, ndim = coords.shape
max_n = 500 # *max iterations* Should be much larger ~1k-10k the low number is just to test.      
        
pos_min = fin_pos_min  # *Posterior min and max*
pos_max = fin_pos_max
pos = [[] for i in range(nwalkers)]
psize = pos_max - pos_min
np.random.seed(42)

for i in range(nwalkers):
    if dpar.mll == True:
        mll_i = dpar.mll_prior[0] + (dpar.comptypes_prior[1]-dpar.comptypes_prior[0]) * np.random.rand()
        
    else:
        mll_i=0
         
    if dpar.amin == True:
        amin_i = dpar.amin_prior[0] + (dpar.amin_prior[1]-dpar.amin_prior[0]) * np.random.rand()
         
    else:
        amin_i=0
     
    if dpar.aexp:
        aexp_i = dpar.aexp_prior[0] + (dpar.aexp_prior[1]-dpar.aexp_prior[0]) * np.random.rand()
         
    else:
        aexp_i=0
     
    if dpar.mexp == True:
        mexp_i = dpar.mexp_prior[0] + (dpar.mexp_prior[1]-dpar.mexp_prior[0]) * np.random.rand()
         
    else:
         mexp_i=0
    
    
    
    if 'Asi' in comptypes_array or 'asi' in comptypes_array:
        asi_i = dpar.comptypes_prior[0] + (dpar.comptypes_prior[1]-dpar.comptypes_prior[0]) * np.random.rand()
        
    else:
       asi_i=0
    
    if 'Ac' in comptypes_array or 'ac' in comptypes_array:
        ac_i = dpar.comptypes_prior[0] + (dpar.comptypes_prior[1]-dpar.comptypes_prior[0]) * np.random.rand()
        
    else:
        ac_i=0
    
    if 'Wi' in comptypes_array or 'wi' in comptypes_array:
        wi_i = dpar.comptypes_prior[0] + (dpar.comptypes_prior[1]-dpar.comptypes_prior[0]) * np.random.rand()
        
    else:
        wi_i=0
    
    if 'Th' in comptypes_array or 'th' in comptypes_array:
        th_i = dpar.comptypes_prior[0] + (dpar.comptypes_prior[1]-dpar.comptypes_prior[0]) * np.random.rand()
        
    else:
        th_i=0
    
    if 'Tr' in comptypes_array or 'tr' in comptypes_array:
        tr_i = dpar.comptypes_prior[0] + (dpar.comptypes_prior[1]-dpar.comptypes_prior[0]) * np.random.rand()
        
    else:
        tr_i=0
    
    if 'Fe' in comptypes_array or 'fe' in comptypes_array:
        fe_i = dpar.comptypes_prior[0] + (dpar.comptypes_prior[1]-dpar.comptypes_prior[0]) * np.random.rand()
        
    else:
        fe_i=0
    
    if 'Orth' in comptypes_array or 'orth' in comptypes_array:
        orth_i = dpar.comptypes_prior[0] + (dpar.comptypes_prior[1]-dpar.comptypes_prior[0]) * np.random.rand()
        
    else:
        orth_i=0
    
    if 'A/WI' in comptypes_array or 'a/wi' in comptypes_array:
        awi_i = dpar.comptypes_prior[0] + (dpar.comptypes_prior[1]-dpar.comptypes_prior[0]) * np.random.rand()
        
    else:
        awi_i=0
    
        
        
    pos_with_zeros = np.array([amin_i, aexp_i, mexp_i, asi_i, ac_i, wi_i, th_i, tr_i, fe_i, orth_i, awi_i, mll_i])
    
    pos_pre_array = filter(None, pos_with_zeros) 
    
    pos[i] = np.array(list(pos_pre_array))

pos = np.array(pos)



        
    # aexp_i = pos_min[1] + psize[1] * np.random.rand()
    # mexp_i = pos_min[2] + psize[2] * np.random.rand()
    #f = np.random.dirichlet([10, 2, 2, 2, 1, 1])    # I don't know if this is the best way, but this gives a reasonable
                                                        # array of walkers for the volume fraction of each component
                                                        # that sums to 1
    # lnf_i = pos_min[9] + psize[9] * np.random.rand()
    #pos[i] = np.array([amin_i, aexp_i, mexp_i, asi_i, ac_i, wii_i, th_i, tr_i, fe_i, orth_i, awi_i, mll_i])
filename = "emcee_Agl_6C_100X10X10.h5"   # *File name to save posteriors* Can be used to continue calculation or make corner plots
backend = emcee.backends.HDFBackend(filename)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# if rank == 0:
#     backend.reset(nwalkers, ndim)  # Comment out this line to continue previous calculation working from a
#                                     # previously saved back end. Otherwise, reset the backend for a new run.
#                                     # The rank == 0 makes sure that the reset only occurs once when using mpi.

# #Do MCMC the pool will have however many processors you specified in the command line
# #mpiexec -n 3 python disk_model_mcmc_wrapper.py
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
    pos, prob, state = sampler.run_mcmc(pos, max_n, progress=True, skip_initial_state_check=True)


    #Save input parameters to plot modeled spectra later, remeber to uncomment out what you need!
initial = h5py.File('shaz_MCMC_Agl_6C_60X6X500_functionalized.hdf5', 'w')
initial.create_dataset('wl', data=wl)
initial.create_dataset('ALL', data=fl)
initial.create_dataset('ERR', data=err)
initial.create_dataset('amax', data=amax)
initial.create_dataset('m_1', data= chem_dict[comptypes_array[0]][1])
initial.create_dataset('m_2', data= chem_dict[comptypes_array[1]][1])
initial.create_dataset('m_3', data= chem_dict[comptypes_array[2]][1])
initial.create_dataset('pixsc', data=pixsc)
initial.create_dataset('npix', data=npix)
initial.create_dataset('zpix', data=zpix)
initial.create_dataset('sigma_up', data=sigma_up)
initial.create_dataset('sigma_y', data=sigma_y)
initial.create_dataset('sigma_d', data=sigma_d)
initial.create_dataset('phang', data=phang)
initial.create_dataset('projected_separation_pts_au', data=phang)
initial.close()
