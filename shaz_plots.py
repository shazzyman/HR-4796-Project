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


# User defined functions
from disk_model_func import disk_model_fun
import disk_parameters as dpar

scatt_type = dpar.scatt_type

def lnlike_new(param_list):
    amin, mexp, f_asi, f_ac, f_wi, lnf = param_list
    ld, ldp, lx, ly, fluxmodel_trash = disk_model_fun(wl, amin, amax, aexp, mexp, f_asi, f_ac, f_wi,
                                                    m_1, m_2, m_3, pixsc, npix, zpix, sigma_up, sigma_y, sigma_d,
                                                    phang, dpar.scatt_type, projected_separation_pts_au)
    
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

def chi_square(param_list):
    amin, mexp, f_1, f_2, f_3, lnf = param_list
    ld, ldp, lx, ly, fluxmodel_trash = disk_model_fun(wl, amin, amax, aexp, mexp, f_1, f_2, f_3,
                                                    m_1, m_2, m_3, pixsc, npix, zpix, sigma_up, sigma_y, sigma_d,
                                                    phang, dpar.scatt_type, projected_separation_pts_au)
    
    ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
    ldp_rotated[np.where(ldp_rotated< 0)] = 0
    
    print(mexp)
    fluxmodel = np.zeros(nwl)
    for i in range(0, nwl):
        aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y)
        fluxmodel[i] = medians
        
        
    return (np.sum((fluxmeasured-fluxmodel)**2/(fluxerr **2))), fluxmodel

aexp = -3.0
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

# # Corner plot (10 parameters)
# #emcee_filename = "shaz_emcee_Mie_6C_32X10X100.h5"
# reader = emcee.backends.HDFBackend('shaz_emcee_Agl_6C_100X6X1000.h5', read_only=True)
# tau = reader.get_autocorr_time(tol=0)
# burnin = int(2*np.max(tau))
# thin = int(0.5*np.min(tau))
# samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
# samples2 = reader.get_chain(flat=True)
# samples3=reader.get_chain(flat=False)

# log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
# log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

# m_amin, m_aexp, m_mexp, mfsi, mfac, mfwi, mfth, mftr, mffe, mll= np.percentile(samples2, 50, axis=0)

# plt.rcParams.update({'font.size': 10})
# fig1 = corner.corner(samples2, labels=["$a_{min}$", "$a_{exp}$", "$m_{exp}$", "$f_{asi}$", "$f_{ac}$", "$f_{wi}$", "$f_{th}$", "$f_{tr}$", "$f_{fe}$", "$m_{ll}$"],
#                       range=[[0.01, 4.], [-5., -1.], [27., 33.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [-0.1, 0.1]]
#                       , truths=[m_amin, m_aexp, m_mexp, mfsi, mfac, mfwi, mfth, mftr, mffe, mll])
# fig1.set_size_inches(15, 15)
# plt.show()

# labels=["$a_{min}$", "$a_{exp}$", "$m_{exp}$", "$f_{asi}$", "$f_{ac}$", "$f_{wi}$", "$f_{th}$", "$f_{tr}$", "$f_{fe}$", "$m_{ll}$"]
# for i in range(10):
#     mcmcp = np.percentile(samples[:, i], [16, 50, 84])
#     q = np.diff(mcmcp)
#     txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
#     txt = txt.format(mcmcp[1], q[0], q[1], labels[i])
#     display(Math(txt))

#Corner plot (6 parameters)
emcee_filename = "emcee_Agl_6C_100X10X10.h5"
reader = emcee.backends.HDFBackend(emcee_filename, read_only=True)
tau = reader.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
samples2 = reader.get_chain(flat=True)
samples3=reader.get_chain(flat=False)

log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

amin, mexp, f_1, f_2, f_3, mll= np.percentile(samples2, 50, axis=0)
print('amin = ', amin)
print('mexp = ', mexp)
print('f_si = ', f_1)
print('f_ac = ', f_2)
print('f_wi = ', f_3)
print('mll =', mll)



plt.rcParams.update({'font.size': 10})
fig1 = corner.corner(samples2, labels=["$a_{min}$",  "$m_{exp}$", "$f_{asi}$", "$f_{ac}$", "$f_{wi}$", "$m_{ll}$"],
                      range=[[2, 10.], [30, 34], [0., 1.], [0., 1.], [0., 1.],[-0.1, 0.1]]
                      , truths=[amin, mexp, f_1, f_2, f_3, mll])
fig1.set_size_inches(15, 15)
plt.show()

labels=["$a_{min}$", "$m_{exp}$", "$f_{asi}$", "$f_{ac}$", "$f_{wi}$", "$m_{ll}$"]
for i in range(6):
    mcmcp = np.percentile(samples[:, i], [16, 50, 84])
    q = np.diff(mcmcp)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmcp[1], q[0], q[1], labels[i])
    display(Math(txt))

# #Corner plot (7 parameters)
# #emcee_filename = "shaz_emcee_Mie_6C_32X10X100.h5"
# reader = emcee.backends.HDFBackend('7p_shaz_emcee_Agl_6C_60X7X500_Astro_silicate_Draine_2003.h5', read_only=True)
# tau = reader.get_autocorr_time(tol=0)
# burnin = int(2*np.max(tau))
# thin = int(0.5*np.min(tau))
# samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
# samples2 = reader.get_chain(flat=True)
# samples3=reader.get_chain(flat=False)

# log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
# log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

# m_amin, m_aexp, m_mexp, mfsi, mfac, mfwi, mll= np.percentile(samples2, 50, axis=0)
# print('amin = ', m_amin)
# print('m_aexp =', m_aexp)
# print('mexp = ', m_mexp)
# print('f_si = ', mfsi)
# print('f_ac = ', mfac)
# print('f_wi = ', mfwi)



# plt.rcParams.update({'font.size': 10})
# fig1 = corner.corner(samples2, labels=["$a_{min}$", "$a_{exp}$", "$m_{exp}$", "$f_{asi}$", "$f_{ac}$", "$f_{wi}$", "$m_{ll}$"],
#                       range=[[2, 10.], [-4, -3], [30, 34], [0., 1.], [0., 1.], [0., 1.],[-0.1, 0.1]]
#                       , truths=[m_amin, m_aexp, m_mexp, mfsi, mfac, mfwi, mll])
# fig1.set_size_inches(15, 15)
# plt.show()

# labels=["$a_{min}$", "$m_{aexp}$", "$m_{exp}$", "$f_{asi}$", "$f_{ac}$", "$f_{wi}$", "$m_{ll}$"]
# for i in range(6):
#     mcmcp = np.percentile(samples[:, i], [16, 50, 84])
#     q = np.diff(mcmcp)
#     txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
#     txt = txt.format(mcmcp[1], q[0], q[1], labels[i])
#     display(Math(txt))
    
#Plot Convergence plots
ndim = 6 #depends on number of free parameters
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ["amin", "mexp"," mfsi", "mfac", "mfwi", "mll"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples3[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples3))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

# Plot Fdisk/F* vs wl
param_filename = 'shaz_MCMC_Agl_6C_60X6X500_functionalized.hdf5'
param_vals = h5py.File(param_filename, 'r', libver='latest', swmr=True)
wl = np.array(param_vals['wl'][:])
amax = np.float(np.array(param_vals['amax']))
m_1 = np.array(param_vals['m_1'][:])
m_2 = np.array(param_vals['m_2'][:])
m_3 = np.array(param_vals['m_3'][:])
pixsc = np.float(np.array(param_vals['pixsc']))
npix = np.float(np.array(param_vals['npix']))
zpix = np.float(np.array(param_vals['zpix']))
sigma_up = np.array(param_vals['sigma_up'][:])
sigma_y = np.array(param_vals['sigma_y'][:])
sigma_d = np.array(param_vals['sigma_d'][:])
phang = np.array(param_vals['phang'][:])
param_vals.close()
projected_separation_pts_au = np.array([75.])

# Run forward model from saved parameters
m_aexp = -3.0
ld, ldp, lx, ly, fluxmodel_trash = \
    disk_model_fun(wl, amin, amax, aexp, mexp, f_1, f_2, f_3, m_1, m_2, m_3, 
                       pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, scatt_type, projected_separation_pts_au)


nwl = 11       # Number of wavelengths
fluxmodel = np.zeros(nwl)
for i in range(0, nwl):
    ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
    ldp_rotated[np.where(ldp_rotated< 0)] = 0
    aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y) #calculating medians, these apertures matched the ones in Rodigas Paper
    fluxmodel[i] = medians #inputting medians into array that can be used as fluxmodel
# *Import disk data*
disk_dir = './DiskModelShare'
wl, fl, err = np.loadtxt('Rodigas_export_new.csv', skiprows = 1, unpack=True, delimiter = ',', usecols=(1,2,6))
# Put fluxes and error each into one array that MCMC can fit
fluxmeasured = fl
fluxerr = err
projected_separation_pts_au = np.array([75.]) # *Choose projected separations to fit (in AU)*

# =============================================================================
#Check value of chi square likelihood func
param_list = (amin, mexp, f_1, f_2, f_3, mll)
# # chi_resluts = chi_square(param_list)
# print('likelihood results are:', chi_resluts)

# =============================================================================


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

hdr.insert('SCATT', ('AMAX', amax), after = 'SCATT')
hdr.insert('SCATT', ('AMIN', amin), after = 'AMAX')

fits.writeto('flx_data.fits', data, hdr, overwrite=True)
print(repr(hdr)) 

# Run aperture photometry
pixsc = 1. / (dpar.px_size * dpar.dsun)
arc_to_px = 1. / dpar.px_size




# Plot modeled + measured spectra
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

# Plot scattered light image
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



#This code plots both samples from the emcee and the 50th percentile model
# =============================================================================
# plt.rcParams.update({'font.size': 16})
# fig2 = plt.figure(figsize=(10,8), dpi=200)
# ax = plt.subplot(111)
# 
# inds = np.random.randint(8000, high = 8100, size = 100)  
# for ind in inds:
#     sample = samples[ind]
#     print(sample.shape)
#     m_amin, m_mexp, mfsi, mfac, mfwi, mll = sample
#    
#     m_aexp = -3.0
#     ld, ldp, lx, ly, fluxmodel_trash = \
#         disk_model_fun_new(wl, m_amin, amax, m_aexp, m_mexp, mfsi, mfac, mfwi,  m_asi, m_ac, m_wi,
#                         pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, 'Agglomerate', projected_separation_pts_au)
# 
#     nwl = 11       # Number of wavelengths
#     fluxmodels = np.zeros(nwl)
#     for i in range(0, nwl):
#         ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
#         ldp_rotated[np.where(ldp_rotated< 0)] = 0
#         aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y) #calculating medians, these apertures matched the ones in Rodigas Paper
#         fluxmodels[i] = medians #inputting medians into array that can be used as fluxmodel
#        
#     # Plot modeled + measured spectra
#     plt.semilogy(wl, fluxmodels, color="C1", lw=.5, alpha=0.5)
# 



# m_amin, m_mexp, mfsi, mfac, mfwi, mll= np.percentile(samples, 50, axis=0)
# m_aexp = -3.0
# ld, ldp, lx, ly, fluxmodel_trash = \
#     disk_model_fun_new(wl, m_amin, amax, m_aexp, m_mexp, mfsi, mfac, mfwi,  m_asi, m_ac, m_wi,
#                     pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, 'Agglomerate', projected_separation_pts_au)

# nwl = 11       # Number of wavelengths
# fluxmodel = np.zeros(nwl)
# for i in range(0, nwl):
#     ldp_rotated = rotate(ldp, angle=dpar.pphi, reshape=False)
#     ldp_rotated[np.where(ldp_rotated< 0)] = 0
#     aperture, phot_tables, medians = square_aperture_calcs(ldp_rotated[:, :, i],  sq_ap.w, sq_ap.h, sq_ap.x, sq_ap.y) #calculating medians, these apertures matched the ones in Rodigas Paper
#     fluxmodel[i] = medians #inputting medians into array that can be used as fluxmodel

# plt.semilogy(wl, fluxmodel, color="darkviolet", lw=4, alpha=1, label = "median solution")
# plt.plot(wl[0], fl[0], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[0], fl[0], yerr=err[0], color='black')
# plt.plot(wl[1], fl[1], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[1], fl[1], yerr=err[1], color='black')
# plt.plot(wl[2], fl[2], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[2], fl[2], yerr=err[2], color='black')
# plt.plot(wl[3], fl[3], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[3], fl[3], yerr=err[3], color='black')
# plt.plot(wl[4], fl[4], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[4], fl[4], yerr=err[4], color='black')
# plt.plot(wl[5], fl[5], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[5], fl[5], yerr=err[5], color='black')
# plt.plot(wl[6], fl[6], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[6], fl[6], yerr=err[6], color='black')
# plt.plot(wl[7], fl[7], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[7], fl[7], yerr=err[7], color='black')
# plt.plot(wl[8], fl[8], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[8], fl[8], yerr=err[8], color='black')
# plt.plot(wl[9], fl[9], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[9], fl[9], yerr=err[9], color='black')
# plt.plot(wl[10], fl[10], marker='o', markersize=15, markerfacecolor='black')
# plt.errorbar(wl[10], fl[10], yerr=err[10], color='black')
# plt.xlabel('wavelength $\\mu$m', fontsize=30)
# plt.ylabel('F$_{disk}$ / F$_*$', fontsize=34)
# plt.title("Spectra Fit with " +dpar.scatt_type)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
# plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=14)
# plt.tight_layout()
# plt.subplots_adjust(top=0.88)
       
# plt.show()
# =============================================================================
