# Imports
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
from dictionary import nk_dict


ip = IPython.core.getipython.get_ipython()

#User made functions!
from disk_model_func import disk_model_fun
import disk_parameters as dpar
from getstellarmodel import getstellarmodel_fun
import shared_constants as co

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
#   Minimum grain size between 0.01 and 1 micron
#   Power law grain size distribution exponent between -5 and -1
#   Total disk mass order of magnitude between 22 and 26 (kg)
#   Volume fraction of each compositional component total to 1

# def lnprior(param_list):
#     amin, aexp, mexp, f_1, f_2, f_3, f_4, f_5, f_6, lnf = param_list
#     if 0.01 < amin < 1. and -5. < aexp < -1. and 22. < mexp < 26. and 0. < f_1 < 1. and 0. < f_2 < 1. \
#             and 0. < f_3 < 1. and 0. < f_4 < 1. and 0. < f_5 < 1. and 0. < f_6 < 1. \
#             and 0.99 < f_1 + f_2 + f_3 + f_4 + f_5 + f_6 < 1.01 and -10.0 < lnf < 1.0:
#         return 0.0
#     return -np.inf

# =============================================================================
# # Evaluate the Chi-square likelihood
# def lnlike(param_list):
#     amin, aexp, mexp, f_asi, f_ac, f_wi, f_th, f_tr, f_fe, lnf = param_list
#     ld, ldp, lx, ly, fluxmodel = disk_model_fun(wl, amin, amax, aexp, mexp, f_asi, f_ac, f_wi,
#                                                     f_th, f_tr, f_fe, m_asi, m_ac, m_wi, m_th, m_tr,
#                                                     m_fe, pixsc, npix, zpix, sigma_up, sigma_y, sigma_d,
#                                                     phang, scatt_type, projected_separation_pts_au)
#     inv_sigma2 = 1.0 / (fluxerr ** 2 + fluxmodel ** 2 * np.exp(2 * lnf))
#     return -0.5*(np.sum((fluxmeasured-fluxmodel)**2*inv_sigma2 - np.log(inv_sigma2)))
# 
# def lnprob(param_list):
#     lp = lnprior(param_list)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + lnlike(param_list)
# =============================================================================

global wl, amax, m_asi, m_ac, m_wi, m_th, m_tr, m_fe, projected_separation_pts_au, fluxmeasured, fluxerr

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

# *Import disk data*
disk_dir = './'
wl, fl, err = np.loadtxt(disk_dir+'/Rodigas_table2.txt', skiprows=1,unpack=True, delimiter=',', usecols=(0,2,3))

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

def calc_m_val(chem_type):
    if chem_type == ('Astrosilicate' or 'astrosilicate'):
        nk_file_asi = nk_dict[0]
        nkdat_asi = np.loadtxt(nk_file_asi, skiprows=5, unpack=True)
        wlnk_asi = np.flipud(nkdat_asi[0, :])
        nf_asi = np.flipud(1 + nkdat_asi[3, :])
        kf_asi = np.flipud(nkdat_asi[4, :])
        n_asi = np.interp(wl, wlnk_asi, nf_asi)
        k_asi = np.interp(wl, wlnk_asi, kf_asi)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_asi,k_asi))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_asi - 1j*k_asi
    elif chem_type == ('Amorphous Carbon' or 'amorphous carbon'):
        nk_file_ac = nk_dict[1]
        wlnk_ac, nf_ac, kf_ac = np.loadtxt(nk_file_ac, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_ac = np.interp(wl, wlnk_ac, nf_ac)
        k_ac = np.interp(wl, wlnk_ac, kf_ac)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_ac,k_ac))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_ac - 1j*k_ac
    elif chem_type == ('Water Ice' or 'water ice'):
        nk_file_wi = nk_dict[2]
        wlnk_wi, nf_wi, kf_wi = np.loadtxt(nk_file_wi, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_wi = np.interp(wl, wlnk_wi, nf_wi)
        k_wi = np.interp(wl, wlnk_wi, kf_wi)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_wi,k_wi))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_wi - 1j*k_wi
        
    elif chem_type == ('Tholins' or 'tholins'):
        # Tholins
        nk_file_th = nk_dict[3]
        wlnk_th, nf_th, kf_th = np.loadtxt(nk_file_th, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_th = np.interp(wl, wlnk_th, nf_th)
        k_th = np.interp(wl, wlnk_th, kf_th)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_th,k_th))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_th - 1j*k_th
            
    elif chem_type == ('Troilite' or 'troilite'):
        nk_file_tr = './optical_constants/troilite.txt'
        wlnk_tr, nf_tr, kf_tr = np.loadtxt(nk_file_tr, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_tr = np.interp(wl, wlnk_tr, nf_tr)
        k_tr = np.interp(wl, wlnk_tr, kf_tr)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_tr,k_tr))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_tr - 1j*k_tr
    
    elif chem_type == ('Iron' or 'iron'):
        nk_file_fe = './optical_constants/iron_Henning.txt'
        wlnk_fe, nf_fe, kf_fe = np.loadtxt(nk_file_fe, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_fe = np.interp(wl, wlnk_fe, nf_fe)
        k_fe = np.interp(wl, wlnk_fe, kf_fe)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_fe,k_fe))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_fe - 1j*k_fe
    else:
        raise ValueError('Chemical type must be "Astrosilicate", "Amorphous Carbon", or "Water Ice"')
        
    
    
    return m_val


chem_type1 = 'Astrosilicate'
chem_type2 = 'Amorphous Carbon'
chem_type3 = 'Water Ice'
chem_type4 = 'Tholins'
chem_type5 = 'Troilite'
chem_type6 = 'Iron'

m_1 = calc_m_val(chem_type1)
m_2 = calc_m_val(chem_type2)
m_3 = calc_m_val(chem_type3)
m_4 = calc_m_val(chem_type4)
m_5 = calc_m_val(chem_type5)
m_6 = calc_m_val(chem_type6)

#Put fluxes and error each into one array
fluxmeasured = fl
fluxerr = err
projected_separation_pts_au = np.array([75.]) #Choose projected seperations to fit (in Au)*

#Create 3D array with the particle surface density
pixsc = 1. / dpar.pxs # pixels per AU
npix = np.int(np.round(pixsc *350)) # *Choose number of pixels in model (in plane of disk)*
zpix = np.int(np.round(pixsc * 10)) # *Choose number of pixels in model (vertical)*
sigma_up, sigma_y, sigma_d, phang = calc_phase_ang_surf_dens(pixsc, npix, zpix)

# =============================================================================
# print(sigma_up)
# print(sigma_y)
# print(sigma_d)
# print(phang)
# 
# =============================================================================
# =============================================================================
# sigma_up = (10. ** 24) * sigma_up / np.sum(sigma_up)
# sigma_num = sigma_up / np.trapz((4. / 3.) * np.pi * (2.5 * 1.0 * 3.3 * ((a * 1e-4) ** 3) * abin, a))
# print(sigma_num)
# 
# =============================================================================

# Plot fdisk/F* vs wl parameters, however, not worried abt this plot only want values to aid with forward model
# =============================================================================
# param_filename = 'MCMC_test.hdf5'
# param_vals = h5py.File(param_filename, 'r', libver='latest', swmr=True)
# wl = np.array(param_vals['wl'][:])
# ALL = np.array(param_vals['ALL'][:])
# ERR = np.array(param_vals['ERR'][:])
# 
# amax = np.float(np.array(param_vals['amax']))
# m_asi = np.array(param_vals['m_asi'][:])1
# m_ac = np.array(param_vals['m_ac'][:])
# m_wi = np.array(param_vals['m_wi'][:])
# m_th = np.array(param_vals['m_th'][:])
# m_tr = np.array(param_vals['m_tr'][:])
# m_fe = np.array(param_vals['m_fe'][:])
# pixsc = np.float(np.array(param_vals['pixsc']))
# npix = np.float(np.array(param_vals['npix']))
# zpix = np.float(np.array(param_vals['zpix']))
# sigma_up = np.array(param_vals['sigma_up'][:])
# sigma_y = np.array(param_vals['sigma_y'][:])
# sigma_d = np.array(param_vals['sigma_d'][:])
# phang = np.array(param_vals['phang'][:])
# param_vals.close()
# projected_separation_pts_au = np.array([75.])
# =============================================================================

amin=2.0
amax=100.
aexp= -3.7
mexp=32.0


mfsi=1.
mfac=0.
mfwi=0.
mfth=0.
mftr=0.
mffe=0.


# Run forward model from saved parameters

ld_agl, ldp_agl, lx, ly, fluxmodel = \
    disk_model_fun(wl, amin, amax, aexp, mexp, mfsi, mfac, mfwi, mfth, mftr, mffe, m_1, m_2, m_3, m_4, m_5, m_6,
                    pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, scatt_type, projected_separation_pts_au)
    

ldp_agl_rotated = rotate(ldp_agl, angle=dpar.pphi, reshape=False)
ldp_agl_rotated[np.where(ldp_agl_rotated< 0)] = 0

plt.rcParams.update({'font.size': 16})

fig4 = plt.figure(figsize=(8,6), dpi=200)
ax = fig4.add_subplot(1, 1, 1)
L_mieplt = plt.imshow(10000*ldp_agl_rotated[:, :, 0], norm=colors.PowerNorm(gamma=1./4.), cmap=plt.cm.get_cmap('inferno'), extent=(-1*np.max(lx)/2, np.max(lx)/2, -1*np.max(ly)/2, np.max(ly)/2))
plt.plot([0],[0], marker="*", color='y')
cbar = fig4.colorbar(L_mieplt)
cbar.ax.set_ylabel('F$_{disk}$ / F$_*$ x $10^5$')
ts = ax.transData
coords = [0, 0]
tr = trf.Affine2D().rotate_deg_around(coords[0], coords[1], dpar.pphi)
t = tr + ts

plt.xlabel('$\\delta$ arcsec')
plt.ylabel('$\\delta$ arcsec')
plt.title('  amax ='+str(amax)+  '  amin='+str(amin)+  '  m_aexp='+str(m_aexp)+  '  m_mexp='+str(m_mexp)+  '  mfsi='+str(mfsi)+  '  mfac='+str(mfac)+  '  mfwi='+str(mfwi)+  '  mfth='+str(mfth)+  '  mftr='+str(mftr)+  '  mffe='+str(mffe), fontsize=15, pad=50)
ax.tick_params(labelsize=15)
plt.show()



old_data = ldp_agl_rotated
print(np.isfortran(old_data))
print(np.shape(old_data))

data = np.transpose(old_data, (2, 0, 1))
print(np.shape(data))


primary_hdu = fits.PrimaryHDU(data)
hdul = fits.HDUList([primary_hdu])

hdr = hdul[0].header
hdr['SCATT'] = (scatt_type)

hdr.insert('SCATT', ('AMAX', amax), after = 'SCATT')
hdr.insert('SCATT', ('AMIN', amin), after = 'AMAX')
# =============================================================================
# hdr.insert('EXTEND', ('M_AEXP', m_aexp), after = 'AMIN')
# hdr.insert('EXTEND', ('M_MEXP', m_mexp), after = 'M_AEXP')
# hdr.insert('EXTEND', ('MFSI', mfsi), after = 'M_MEXP')
# hdr.insert('EXTEND', ('MFAC', mfac), after = 'MFSI')
# hdr.insert('EXTEND', ('MFWI', mfwi), after = 'MFAC')
# hdr.insert('EXTEND', ('MFTH', mfth), after = 'MFAC')
# hdr.insert('EXTEND', ('MFTR', mftr), after = 'MFTH')
# hdr.insert('EXTEND', ('MFFE', mffe), after = 'MFTR')
# hdr.insert('EXTEND', ('RMIN', dpar.rmin))
# hdr.insert('EXTEND', ('RMAX', dpar.rmax))
# hdr.insert('EXTEND', ('RMAXHALO', dpar.rmaxhalo))
# hdr.insert('EXTEND', ('XSIGMA', dpar.xsigma))
# hdr.insert('EXTEND', ('XSIGHALO', dpar.xsigmahalo))
# hdr.insert('EXTEND', ('OPHI', dpar.ophi))
# hdr.insert('EXTEND', ('PPHI', dpar.pphi))
# =============================================================================

fits.writeto('flx_data.fits', data, hdr, overwrite=True)


print(repr(hdr)) 

pixsc = 1. / (dpar.px_size * dpar.dsun)
arc_to_px = 1. / dpar.px_size

# w =  0.1523 * arc_to_px #STIS aperture size ["]
w = 0.2262 * arc_to_px #NICMOS aperture size ["]
# w = 0.111 * arc_to_px #Clio-2 aperture size ["]
h = w

x = 82.5 + (dpar.rmin + dpar.rmax) * pixsc / 2 * np.sin( 206.8 * np.pi / 180) #xvalue of position
y = 82.5 + (dpar.rmin + dpar.rmax) * pixsc / 2 * np.cos( 206.8 * np.pi / 180) #yvalue of position

#Calculate median flux at different wavelengths
#aperture, phot_tables, medians = square_aperture_calcs(spec_wl_data,  w, h, x, y)

spec_wl_data1 = data[0, :, :]
aperture, phot_tables, medians = square_aperture_calcs(spec_wl_data1,  w, h, x, y)
median1 = medians


