import numpy as np
import numba as nb
import h5py
from scipy import optimize
from scipy.ndimage.interpolation import rotate
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.transforms as trf
from IPython.display import display, Math
import IPython
from square_apertures import square_aperture_calcs
from dictionary import chem_dict
from disk_parameters import comptypes_array
import square_apertures as sq_ap
from square_apertures import square_aperture_calcs

ip = IPython.core.getipython.get_ipython()

#User made functions!
from untitled2 import disk_model_fun
import disk_parameters as dpar
from getstellarmodel import getstellarmodel_fun
import shared_constants as co

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

def lnprior(param_list):
    
        
    comptypes_array = dpar.comptypes_array   
    
    amin, aexp, mexp, f_1, f_2, f_3, mll = param_list
    
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


global wl, amax, projected_separation_pts_au, fluxmeasured, fluxerr

def bruggeman2(eff, *args):
    effc = eff[0:eff.size//2] + 1j*eff[eff.size//2:]
    e1=args[0]+1j*args[1]
    e2=(1.0 + np.zeros(e1.size)) + 1j*(0.0 + np.zeros(e1.size))
    minc = 0.25 * (e1 - effc) / (e1 + 2 * effc) + 0.75 * (e2 - effc) / (e2 + 2 * effc)
    return np.concatenate((np.real(minc), np.imag(minc)))

if dpar.amin == False:
    amin = 3.0
    
if dpar.aexp == False:
    aexp = -3.
    
if dpar.mexp == False:
    mexp = 3.
    
if dpar.mll == False:
    mll = 0.

m_dict = []

for comptype in comptypes_array:
    m_dict = np.append(m_dict, chem_dict[comptype][1])
    
# Set wavelength range and maximum grain size   *Edit for specific disk*
wlo = 0.5737    # Minimum wavelength
whi = 3.7565    # Maximum wavelength
nwl = 11       # Number of wavelengths
wl = np.linspace(wlo, whi, num=nwl)
amax = 100.

# *Import disk data*
disk_dir = './DiskModelShare'
wl, fl, err = np.loadtxt('Rodigas_export_new.csv', skiprows = 1, unpack=True, delimiter = ',', usecols=(1,2,6))

    
    
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
    
# Put fluxes and error each into one array that MCMC can fit
fluxmeasured = fl
fluxerr = err
projected_separation_pts_au = np.array([75.]) # *Choose projected separations to fit (in AU)*

# Create 3D array with particle surface density
pixsc = 1. / dpar.pxs  # *pixels per AU*
npix = int(np.round(pixsc * 350))    # *Choose number of pixels in model (in plane of disk)*
zpix = int(np.round(pixsc * 10))     # *Choose number of pixels in model (vertical)*
sigma_up, sigma_y, sigma_d, phang = calc_phase_ang_surf_dens(pixsc, npix, zpix)

amin=2.0
amax=100.
aexp= -3.7
mexp=32.0
f_1=1.
f_2=0.
f_3=0.
mll = 0

m_1 = chem_dict[comptypes_array[0]][1]
m_2 = chem_dict[comptypes_array[1]][1]
m_3 = chem_dict[comptypes_array[2]][1]


ld, ldp, lx, ly, fluxmodel_trash = disk_model_fun(wl, amin, amax, aexp, mexp, f_1, f_2, f_3, m_1, m_2, m_3, 
                   pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, scatt_type, projected_separation_pts_au)
    
ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
ldp_rotated[np.where(ldp_rotated< 0)] = 0

    
fluxmodel = np.zeros(nwl)
for i in range(0, nwl):
        aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  
                                        sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y) #calculating medians, these apertures matched the ones in Rodigas Paper
        fluxmodel[i] = medians #inputting medians into array that can be used as fluxmodel
    
# Plot scattered light image
plt.rcParams.update({'font.size': 16})
bar_width = dpar.fid_bar
bar_height = 2.0
fig4 = plt.figure(figsize=(8,6), dpi=200)
ax = fig4.add_subplot(1, 1, 1)
L_mieplt = plt.imshow(10000*ldp_rotated[:, :, 0], norm=colors.PowerNorm(gamma=1./4.), 
                      cmap=plt.cm.get_cmap('inferno'), extent=(-1*np.max(lx)/2, 
                                    np.max(lx)/2, -1*np.max(ly)/2, np.max(ly)/2))
plt.plot([0],[0], marker="*", color='y')
cbar = fig4.colorbar(L_mieplt)
cbar.ax.set_ylabel('F$_{disk}$ / F$_*$ x $10^5$')
ts = ax.transData
coords = [0, 0]
tr = trf.Affine2D().rotate_deg_around(coords[0], coords[1], dpar.pphi)
t = tr + ts
#rect = Rectangle((0 - bar_height/2, 0 - bar_width/2), bar_height, bar_width, color='k', alpha=1.0, transform=t)
#ax.add_patch(rect)
plt.xlabel('$\\delta$ arcsec')
plt.ylabel('$\\delta$ arcsec')
#plt.title('  amax ='+str(amax)+  '  amin='+str(m_amin)+  '  m_aexp='+str(m_aexp)+  '  m_mexp='+str(m_mexp)+  '  mfsi='+str(mfsi)+  '  mfac='+str(mfac)+  '  mfwi='+str(mfwi)+  '  mfth='+str(mfth)+  '  mftr='+str(mftr)+  '  mffe='+str(mffe), fontsize=15, pad=50)

plt.show()