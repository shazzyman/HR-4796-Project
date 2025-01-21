#pip #!/usr/bin/env python

# Imports
import numpy as np
import corner
import emcee
import h5py
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import matplotlib.transforms as trf
from IPython.display import display, Math
from astropy.io import fits
import IPython
from square_apertures import square_aperture_calcs
ip = IPython.core.getipython.get_ipython()
import square_apertures as sq_ap
import disk_parameters as dpar

from dictionary import chem_dict


if dpar.amin == False:
    amin = 3.0
    
if dpar.aexp == False:
    aexp = -3.
    
if dpar.mexp == False:
    mexp = 3.
    
if dpar.mll == False:
    mll = 0.
    
wlo = 0.5737    # Minimum wavelength
whi = 3.7565    # Maximum wavelength
nwl = 11       # Number of wavelengths
wl = np.linspace(wlo, whi, num=nwl)
amax = 100.

# User defined functions

from disk_model_func import disk_model_fun
import disk_parameters as dpar
from disk_model_func import param_list_final
from disk_parameters import comptypes_array


def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += emcee.autocorr.function_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def free_vars(emcee_filename):
    reader = emcee.backends.HDFBackend(emcee_filename, read_only=True)
    tau = reader.get_autocorr_time(tol=0)
    burnin = int(2*np.max(tau))
    thin = int(0.5*np.min(tau))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    samples2 = reader.get_chain(flat=True)
    samples3=reader.get_chain(flat=False)

    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
    log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

    percentiles = []
    for variable in param_list_final:
        
        percentile = np.percentile(samples2, 50, axis=0)
        percentiles = np.append(percentiles, percentile)
        
    
        
    return samples, samples2, samples3, burnin, thin, log_prob_samples, log_prior_samples

#Def Corner plot Function
# def corner_plot(final_param_list, samples2, samples):
#     labels = ["$"+variable+"$" for variable in param_list_final]
#     fig1 = corner.corner(samples2, labels=labels,
#                          range=[[2, 10.], [30, 34], [0., 1.], [0., 1.], [0., 1.],[-0.1, 0.1]]
#                          , truths=[variable])
#     fig1.set_size_inches(15, 15)
#     plt.show()
    
def corner_plot(final_param_list, samples2, samples):

    plt.rcParams.update({'font.size': 10})
    labels = ["$"+variable+"$" for variable in param_list_final]
    fig1 = corner.corner(samples2, labels=labels,
                         range=[[2, 10.], [30, 34], [0., 1.], [0., 1.], [0., 1.],[-0.1, 0.1]]
                         , truths=[variable for variable in param_list_final])
    fig1.set_size_inches(15, 15)
    plt.show()

    for variable in param_list_final:
        labels=["$"+variable+"$"]
        for i in range(0, len(param_list_final)):
            mcmcp = np.percentile(samples[:, i], [16, 50, 84])
            q = np.diff(mcmcp)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmcp[1], q[0], q[1], labels[i])
            display(Math(txt))
        
       
    return plt.show()

#Def Convergence Plot Function
def conv_plot(param_list_final, samples3):
    ndim = len(param_list_final)
    for variable in param_list_final:
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        labels = ["$" + variable +"$"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples3[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples3))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)   
            
   
    return fig, axes


   
# Plot Fdisk/F* vs wl Function
def spec_fit(param_list_final, param_filename):

    
    param_vals = h5py.File(param_filename, 'r', libver='latest', swmr=True)
    wl = np.array(param_vals['wl'][:])
    amax = np.float(np.array(param_vals['amax']))
    m_dict = []
    for comptype in comptypes_array:
        m_dict = np.append(m_dict, chem_dict[comptype][1])
        

    pixsc = np.float(np.array(param_vals['pixsc']))
    npix = np.float(np.array(param_vals['npix']))
    zpix = np.float(np.array(param_vals['zpix']))
    sigma_up = np.array(param_vals['sigma_up'][:])
    sigma_y = np.array(param_vals['sigma_y'][:])
    sigma_d = np.array(param_vals['sigma_d'][:])
    phang = np.array(param_vals['phang'][:])
    param_vals.close()
    projected_separation_pts_au = np.array([75.])
    
    
    ld, ldp, lx, ly, fluxmodel_trash = \
        disk_model_fun(wl, amin, amax, aexp, mexp, comptypes_array, 
                           pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, dpar.scatt_type, projected_separation_pts_au)
    fluxmodel = np.zeros(nwl)
    for i in range(0, nwl):
        ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
        ldp_rotated[np.where(ldp_rotated< 0)] = 0
        aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y) #calculating medians, these apertures matched the ones in Rodigas Paper
        fluxmodel[i] = medians #inputting medians into array that can be used as fluxmodel
    
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
    plt.xlabel('wavelength $\\mu$m', fontsize=34)
    plt.ylabel('F$_{disk}$ / F$_*$', fontsize=30)
    plt.title("Spectra Fit with " +dpar.scatt_type)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    
    plt.rcParams.update({'font.size': 16})
    
    # Set up data and fits file
    ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
    ldp_rotated[np.where(ldp_rotated< 0)] = 0
    
    old_data = ldp_rotated
    print(np.isfortran(old_data))
    print(np.shape(old_data))
    
    data = np.transpose(old_data, (2, 0, 1))
    print(np.shape(data))
    
    
    primary_hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([primary_hdu])
    
    hdr = hdul[0].header
    hdr['SCATT'] = ('Agl')
    
    
    fits.writeto('flx_data.fits', data, hdr, overwrite=True)
    print(repr(hdr)) 
    
    return plt.show(), ldp_rotated, ldp, lx, ly, amax

def spec_image(param_list_final):
    param_vals = h5py.File(param_filename, 'r', libver='latest', swmr=True)
    wl = np.array(param_vals['wl'][:])
    amax = np.float(np.array(param_vals['amax']))
    for comptype in comptypes_array:
        m_ = np.array(param_vals['m_' +comptype][:])
        
    pixsc = np.float(np.array(param_vals['pixsc']))
    npix = np.float(np.array(param_vals['npix']))
    zpix = np.float(np.array(param_vals['zpix']))
    sigma_up = np.array(param_vals['sigma_up'][:])
    sigma_y = np.array(param_vals['sigma_y'][:])
    sigma_d = np.array(param_vals['sigma_d'][:])
    phang = np.array(param_vals['phang'][:])
    param_vals.close()
    projected_separation_pts_au = np.array([75.])
   
    lld, ldp, lx, ly, fluxmodel_trash = \
        disk_model_fun(wl, amin, amax, aexp, mexp, comptypes_array, 
                           pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, dpar.scatt_type, projected_separation_pts_au)
    fluxmodel = np.zeros(nwl)
    for i in range(0, nwl):
        ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
        ldp_rotated[np.where(ldp_rotated< 0)] = 0
        aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y) #calculating medians, these apertures matched the ones in Rodigas Paper
        fluxmodel[i] = medians #inputting medians into array that can be used as fluxmodel
       
       
    plt.rcParams.update({'font.size': 16})
    bar_width = dpar.fid_bar
    bar_height = 2.0
    fig4 = plt.figure(figsize=(8,6), dpi=200)
    ax = fig4.add_subplot(1, 1, 1)
    L_mieplt = plt.imshow(10000*ldp_rotated[:, :, 0], norm=colors.PowerNorm(gamma=1./4.), cmap=plt.cm.get_cmap('inferno'), extent=(-1*np.max(lx)/2, np.max(lx)/2, -1*np.max(ly)/2, np.max(ly)/2))
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
        
    return plt.show()

# *Import disk data*
disk_dir = './DiskModelShare'
wl, fl, err = np.loadtxt('Rodigas_export_new.csv', skiprows = 1, unpack=True, delimiter = ',', usecols=(1,2,6))
# Put fluxes and error each into one array that MCMC can fit
fluxmeasured = fl
fluxerr = err
projected_separation_pts_au = np.array([75.]) # *Choose projected separations to fit (in AU)*
   
# # =============================================================================

nwl = 11    
emcee_filename = "shaz_emcee_Agl_6C_60X6X500_Astro_silicate.h5"
param_filename = 'shaz_MCMC_Agl_6C_60X6X500_Astro_silicate.hdf5'


samples, samples2, samples3, burnin, thin, log_prob_samples, log_prior_samples = free_vars(emcee_filename)

# corner_plt = corner_plot(param_list_final, samples2, samples)

# fig, axes = conv_plot(param_list_final, samples3)

spec_fit = spec_fit(param_list_final, param_filename)

spec_image = spec_image(param_list_final)
# # =============================================================================
