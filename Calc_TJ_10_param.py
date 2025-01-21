```python
"""
This Python script analyzes the spectral properties of debris disks around stars, focusing on 
dust grain modeling, phase angle calculations, and comparison with observational data. It integrates 
several astrophysical tools and libraries, allowing for advanced analysis and visualization. Key components include:

1. **Imports and Dependencies**:
   - Utilizes scientific libraries such as `numpy`, `matplotlib`, and `scipy` for computation and visualization.
   - Leverages `numba` for performance optimization and `astropy` for handling astronomical data.
   - Includes user-defined modules for disk parameters, aperture calculations, and stellar modeling.

2. **Dust Grain Modeling**:
   - Defines refractive indices for multiple dust components (e.g., astrosilicates, amorphous carbon, water ice).
   - Applies Bruggeman mixing rules for porous spheres to calculate composite optical properties.
   - Handles porosity effects and interpolates refractive index data for specific wavelengths.

3. **Phase Angle Calculations**:
   - Computes phase angles and density distributions across 3D disks using a voxel-based approach.
   - Differentiates between disk and halo regions based on geometric and density thresholds.

4. **Flux and Likelihood Calculations**:
   - Implements a forward model to compute disk flux using Mie or agglomerate scattering models.
   - Evaluates the likelihood function (`lnlike`) for fitting model fluxes to observational data.
   - Incorporates observational uncertainties into the likelihood evaluation.

5. **Visualization**:
   - Plots the modeled and observed spectra with error bars, allowing visual comparison.
   - Highlights key spectral features and evaluates model accuracy over multiple test values.

6. **Optimization and Scalability**:
   - Accelerates critical computations with `numba`-optimized functions for phase angle and flux calculations.
   - Uses efficient file I/O via HDF5 for reading optical constants and saving results.

7. **Modular Design**:
   - Separates functionality into reusable components such as `lnlike`, `calc_phase_angle`, and refractive index calculations.
   - Supports customization for specific disk systems by modifying parameters like stellar properties, disk geometry, and dust composition.
"""
```

import numpy as np
from astropy.nddata.blocks import block_reduce
from scipy import interpolate
import numba as nb
# User defined
import disk_parameters_TJCalcs as dpar
from disk_model import disk_model_fun
import square_apertures as sq_ap
from square_apertures import square_aperture_calcs
import shared_constants as co
from scipy import optimize
from getstellarmodel import getstellarmodel_fun
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage.interpolation import rotate
from astropy.io import fits
import matplotlib.transforms as trf
import IPython

ip = IPython.core.getipython.get_ipython()


def lnlike(param_list):
    amin, aexp, mexp, f_asi, f_ac, f_wi, f_th, f_tr, f_fe, lnf = param_list
    ld, ldp, lx, ly, fluxmodel_trash = disk_model_fun(wl, amin, amax, aexp, mexp, f_asi, f_ac, f_wi,
                                                    f_th, f_tr, f_fe, m_asi, m_ac, m_wi, m_th, m_tr,
                                                    m_fe, pixsc, npix, zpix, sigma_up, sigma_y, sigma_d,
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

def lnlike_temp(param_list):
    amin, aexp, mexp, f_asi, f_ac, f_wi, f_th, f_tr, f_fe, lnf = param_list
    ld, ldp, lx, ly, fluxmodel_trash = disk_model_fun(wl, amin, amax, aexp, mexp, f_asi, f_ac, f_wi,
                                                    f_th, f_tr, f_fe, m_asi, m_ac, m_wi, m_th, m_tr,
                                                    m_fe, pixsc, npix, zpix, sigma_up, sigma_y, sigma_d,
                                                    phang, scatt_type, projected_separation_pts_au)
    
    ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
    ldp_rotated[np.where(ldp_rotated< 0)] = 0
    
    print(mexp)
    fluxmodel = np.zeros(nwl)
    for i in range(0, nwl):
        aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y)
        fluxmodel[i] = medians
        
        
    sigma2 = (fluxerr ** 2 + fluxmodel ** 2 * np.exp(2 * lnf))
    #inv_sigma2 = 1.0 / (fluxerr ** 2 * np.exp(2 * lnf))
    # print('the value of inv sig is', inv_sigma2)
    # print('fluxerr is', fluxerr)
    # print('fluxmodel is', fluxmodel)
    # print('lnf is', lnf)
   
    return  (np.sum((fluxmeasured-fluxmodel)**2)) ,fluxmodel
                         #/ sigma2 + np.log(sigma2)))
                         




# Calculate the phase angle of each portion of the disk
@nb.jit(nb.types.Tuple((nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :]))
        (nb.float64, nb.int64, nb.int64), nopython=True)

def calc_phase_angle(pixsc, npix, zpix):
    # Create 3d arrays with particle number density and phase angles
    sigma_up = np.zeros((npix, npix, zpix)) #Number density for each pixel
    sigma_y = np.zeros((npix, npix, zpix)) #Where  each volumetric pixel ends up after projection
    sigma_d = np.zeros((npix, npix, zpix)) #Distance from each voxel to the star in meters
    phang = np.zeros((npix, npix, zpix)) #Phase angle for each volumetric pixel
    
    hz_center_rmax = pixsc * dpar.rmax * np.tan(np.pi * (1. - dpar.ophi / 180)) #height of disk at breakpoint
    sigma_up_rmax_xsigma = ((dpar.rmax) ** dpar.xsigma) / (2 * hz_center_rmax) #Surface density of disk at breakpoint
    sigma_up_rmax_xsigmahalo = ((dpar.rmax) ** dpar.xsigmahalo) / (2 * hz_center_rmax) #Surface density of the halo
    sigma_up_halo_norm = sigma_up_rmax_xsigma/sigma_up_rmax_xsigmahalo 
    for xi in range(0, npix):
        for yi in range(0, npix):
            dxy_center = np.sqrt((xi - npix / 2.) ** 2 + (yi - npix / 2.) ** 2) #Horizontal distance to star
            hz_center = dxy_center * np.tan(np.pi * (1. - dpar.ophi / 180.) / 2) #Total height of the disk at each x, y point
            for zi in range(0, zpix):
                dcenter = np.sqrt((xi - npix / 2.) ** 2 + (yi -npix / 2.) ** 2 + (zi -zpix / 2.) ** 2) #Total distance from the voxel to the star in pixels
                d_y0 = np.sqrt((xi - npix / 2.) ** 2 + yi **2.) #Distance from the bottom of x-y projection
                sigma_d[xi, yi, zi] = (1.0 / pixsc) * dcenter * co.rAU #Total distance from the voxel to the star in meters
                #Below sets Number density and phase angles for the disk
                if ((dxy_center >= pixsc * dpar.rmin) and (dxy_center <= pixsc * dpar.rmax) and (zi >= (zpix / 2. - hz_center)) and (zi <= (zpix / 2. +hz_center))):
                    sigma_y[xi, yi, zi] = (yi - npix / 2.) * dpar.cos_i - (zi - zpix / 2.) * dpar.sin_i + npix / 2.
                    sigma_up[xi, yi, zi] = sigma_up_halo_norm*((dxy_center / pixsc) ** dpar.xsigmahalo) / (2 * hz_center)
                    tht = np.arctan((zi - zpix / 2) / dxy_center)
                    phi = np.arccos(((npix / 2.) ** 2 + dxy_center ** 2 - d_y0 ** 2) / (2 * (npix / 2.) *dxy_center))
                    phang[xi, yi, zi] = np.arccos(dpar.cos_i * np.sin(-tht) + dpar.sin_i * np.cos(-tht) * np.cos(phi))
                #  Below sets number density and phase angles for halo
                if ((dxy_center > pixsc * dpar.rmax) and (dxy_center <= pixsc * dpar.rmaxhalo) and
                        (zi >= (zpix / 4.0 - hz_center)) and (zi <= (zpix / 4.0 + hz_center))):
                    sigma_y[xi, yi, zi] = (yi - npix / 2.) * dpar.cos_i - (zi - zpix / 2.) * dpar.sin_i + npix / 2.
                    sigma_up[xi, yi, zi] = sigma_up_halo_norm*((dxy_center / pixsc) ** dpar.xsigmahalo) / (2 * hz_center)
                    tht = np.arctan((zi - zpix / 2) / dxy_center)
                    phi = np.arccos(((npix / 2.) ** 2 + dxy_center ** 2 - d_y0 ** 2) / (2 * (npix / 2.) * dxy_center))
                    phang[xi, yi, zi] = np.arccos(dpar.cos_i * np.sin(-tht) + dpar.sin_i * np.cos(-tht) * np.cos(phi))
                    
    return sigma_up, sigma_y, sigma_d, phang



def lnprior(param_list):
    amin, aexp, mexp, f_asi, f_ac, f_wi, f_th, f_tr, f_fe, lnf = param_list
    if 0.01 < amin < 1. and -5. < aexp < -1. and 22. < mexp < 26. and 0. < f_asi < 1. and 0. < f_ac < 1. \
            and 0. < f_wi < 1. and 0. < f_th < 1. and 0. < f_tr < 1. and 0. < f_fe < 1. \
            and 0.99 < f_asi + f_ac + f_wi + f_th + f_tr + f_fe < 1.01 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


global wl, amax, m_asi, m_ac, m_wi, m_th, m_tr, m_fe, projected_separation_pts_au, fluxmeasured, fluxerr

# *Import disk data*
disk_dir = './DiskModelShare'
wl, fl, err = np.loadtxt('Rodigas_export_new.csv', skiprows =2, unpack=True, delimiter = ',', usecols=(1,2,6))
# Put fluxes and error each into one array that MCMC can fit
fluxmeasured = fl
fluxerr = err
projected_separation_pts_au = np.array([75.]) # *Choose projected separations to fit (in AU)*


nwl = wl.size
# Bruggeman mixing rule for porous spheres
def bruggeman2(eff, *args):
    effc = eff[0:eff.size//2] + 1j*eff[eff.size//2:]
    e1=args[0]+1j*args[1]
    e2=(1.0 + np.zeros(e1.size)) + 1j*(0.0 + np.zeros(e1.size))
    minc = 0.25 * (e1 - effc) / (e1 + 2 * effc) +0.75 * (e2 - effc) / (e2 + 2 *effc)
    return np.concatenate((np.real(minc), np.imag(minc)))

# Set wavelength range and maximum grain size   *Edit for specific disk*
#wlo = 0.5385    # Minimum wavelength
#whi = 0.9877    # Maximum wavelength
#nwl = 10        # Number of wavelengths
#wl = np.linspace(wlo, whi, num=nwl)

amax = 100.0


calc_stellarmodel = True
if calc_stellarmodel:
    # Stellar parameters    *Edit for specific disk*
    Tstar = 9250     #[K] Stellar temp
    lstar = 21.1     #[Lsun] Stellar Luminosity
    Mstar = 2.5       #[Msun] Stellar mass
    
    #Get stellar model
    Rstar = np.sqrt(lstar*co.lsun/(4.0*np.pi*co.sig*Tstar**4))/co.Rsun
    logg = np.log10(Mstar)-2*np.log10(Rstar) + 4.437
    starmodelfull, wlfull = getstellarmodel_fun(Tstar, logg=logg, modeldir='./stellarmodels/') #Stellar irradiance [Jy/ster]
    starmodel = np.interp(wl, wlfull, starmodelfull)
    Lstar = 4*np.pi*(Rstar**2.0)*starmodel
    
#Load dust refractive indices
porosity = False #True if using porous spheres
scatt_type = dpar.scatt_type 
n0 = 1.0 +np.zeros(wl.size)
k0 = 0.1 + np.zeros(wl.size)
m_guess = np.concatenate((n0, k0)) #Used if 'Porous' chosen
m_brug = np.zeros(m_guess.shape)

#Astrosilicate
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
    
#Amorphous carbon
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
    
#Put fluxes and error each into one array
fluxmeasured = fl
fluxerr = err
projected_separation_pts_au = np.array([75.]) #Choose projected seperations to fit (in Au)*

#Create 3D array with the particle surface density
pixsc = 1. / dpar.pxs # pixels per AU
npix = np.int(np.round(pixsc *350)) # *Choose number of pixels in model (in plane of disk)*
zpix = np.int(np.round(pixsc * 10)) # *Choose number of pixels in model (vertical)*
sigma_up, sigma_y, sigma_d, phang = calc_phase_angle(pixsc, npix, zpix)

# =============================================================================

amin=2.0
amax=100.
m_aexp= -3.7
m_mexp=31.0
mfsi=1.
mfac=0.
mfwi=0.
mfth=0.
mftr=0.
mffe=0.
lnf = 0



#=============================================================================
mexp_test_vals = np.array([ 30., 30.05, 30.1, 30.15, 30.2, 30.25, 30.3, 30.35])
#mexp_test_vals = np.array([ 31.05])
n_mexp = mexp_test_vals.size
mexp_loop = np.zeros(n_mexp)

for i in range(0, n_mexp): 
    param_list = amin, m_aexp, mexp_test_vals[i], mfsi, mfac, mfwi, mfth, mftr, mffe, lnf

    

    mexp_loop[i], fluxmodel = lnlike_temp(param_list)

print(mexp_loop)
#=============================================================================


plt.rcParams.update({'font.size': 16})
fig2 = plt.figure(figsize=(10,8), dpi=200)
ax = plt.subplot(111)
plt.semilogy(wl, fluxmodel, color="darkviolet", lw=2, alpha=0.8, label='Model Flux')
plt.plot(wl[0], fl[0], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[0], fl[0], yerr=err[0], color='black')
plt.plot(wl[1], fl[1], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[1], fl[1], yerr=err[1], color='black')
plt.plot(wl[2], fl[2], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[2], fl[2], yerr=err[2], color='black')
plt.plot(wl[3], fl[3], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[3], fl[3], yerr=err[3], color='black')
plt.plot(wl[4], fl[4], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[4], fl[4], yerr=err[4], color='black')
plt.plot(wl[5], fl[5], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[5], fl[5], yerr=err[5], color='black')
plt.plot(wl[6], fl[6], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[6], fl[6], yerr=err[6], color='black')
plt.plot(wl[7], fl[7], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[7], fl[7], yerr=err[7], color='black')
plt.plot(wl[8], fl[8], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[8], fl[8], yerr=err[8], color='black')
plt.plot(wl[9], fl[9], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[9], fl[9], yerr=err[9], color='black')
plt.plot(wl[10], fl[10], marker='o', markersize=10, markerfacecolor='black')
plt.errorbar(wl[10], fl[10], yerr=err[10], color='black')
plt.xlabel('wavelength $\\mu$m', fontsize=16)
plt.ylabel('F$_{disk}$ / F$_*$', fontsize=16)
plt.title("Spectra Fit with " +dpar.scatt_type)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
