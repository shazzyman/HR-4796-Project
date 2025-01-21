#!/usr/bin/env python

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
import IPython
#import mcmc_Shaz as mcShaz
ip = IPython.core.getipython.get_ipython()

# User defined functions
from disk_model import disk_model_fun
import disk_parameters as dpar

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

# Corner plot
emcee_filename = "shaz_emcee_Agl_6C_60X6X500_Astro_silicate.h5"
reader = emcee.backends.HDFBackend(emcee_filename, read_only=True)
tau = reader.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

m_amin, m_mexp, mfsi, mfac, mfwi, mll= np.percentile(samples, 50, axis=0)

plt.rcParams.update({'font.size': 10})
fig1 = corner.corner(samples, labels=["$a_{min}$", "$a_{exp}$", "$m_{exp}$", "$f_{asi}$", "$f_{ac}$", "$f_{wi}$", "$f_{th}$", "$f_{tr}$", "$f_{fe}$", "$m_{ll}$"],
                     range=[[0.01, 1.], [15., 26.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [-10., 1.0]]
                      , truths=[m_amin, m_mexp, mfsi, mfac, mfwi, mll])
fig1.set_size_inches(15, 15)
plt.show()

labels=["$a_{min}$", "$a_{exp}$", "$m_{exp}$", "$f_{asi}$", "$f_{ac}$", "$f_{wi}$", "$f_{th}$", "$f_{tr}$", "$f_{fe}$", "$m_{ll}$"]
for i in range(10):
    mcmcp = np.percentile(samples[:, i], [16, 50, 84])
    q = np.diff(mcmcp)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmcp[1], q[0], q[1], labels[i])
    display(Math(txt))

# # Plot Fdisk/F* vs wl
# param_filename = 'MCMC_Mie_6C_100X10X10_.hdf5'
# param_vals = h5py.File(param_filename, 'r', libver='latest', swmr=True)
# wl = np.array(param_vals['wl'][:])
# ALL = np.array(param_vals['ALL'][:])
# ERR = np.array(param_vals['ERR'][:])
# #AU20 = np.array(param_vals['AU20'][:])
# #ERR20 = np.array(param_vals['ERR20'][:])
# #AU30 = np.array(param_vals['AU30'][:])
# #ERR30 = np.array(param_vals['ERR30'][:])
# #AU40 = np.array(param_vals['AU40'][:])
# #ERR40 = np.array(param_vals['ERR40'][:])
# amax = np.float(np.array(param_vals['amax']))
# m_asi = np.array(param_vals['m_asi'][:])
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

# # Run forward model from saved parameters
# ld_agl, ldp_agl, lx, ly, fluxmodel = \
#     disk_model_fun(wl, m_amin, amax, m_aexp, m_mexp, mfsi, mfac, mfwi, mfth, mftr, mffe, m_asi, m_ac, m_wi, m_th, m_tr, m_fe,
#                    pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, 'Mie', projected_separation_pts_au)

# #[AU10fit, AU20fit, AU30fit, AU40fit] = np.array_split(fluxmodel, 4)

# # Plot modeled + measured spectra
# plt.rcParams.update({'font.size': 16})
# fig2 = plt.figure(figsize=(10,8), dpi=200)
# ax = plt.subplot(111)
# plt.semilogy(wl, fluxmodel, color="darkviolet", lw=2, alpha=0.8, label='75AU fit')
# plt.semilogy(wl, ALL, color="k", alpha=0.1, label='10 AU measured')

# plt.xlabel('wavelength $\\mu$m', fontsize=16)
# plt.ylabel('F$_{disk}$ / F$_*$', fontsize=16)
# plt.title("Spectra Fit with spheres")
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
# plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=14)
# plt.tight_layout()
# plt.subplots_adjust(top=0.88)
# plt.show()

# ldp_agl_rotated = rotate(ldp_agl, angle=dpar.pphi, reshape=False)
# ldp_agl_rotated[np.where(ldp_agl_rotated< 0)] = 0

# # Flux contribution from a 'slice' of the disk
# fig3 = plt.figure(figsize=(8,6), dpi=200)
# s_up = plt.imshow(ld_agl[:, :, 10, 0], norm=colors.PowerNorm(gamma=1./2.), cmap=plt.cm.get_cmap('inferno'), extent=(-1*np.max(lx)/2, np.max(lx)/2, -1*np.max(ly)/2, np.max(ly)/2))
# plt.xlabel('AU')
# plt.ylabel('AU')
# plt.colorbar()
# plt.show()

# # Plot modeled scattered light image
# plt.rcParams.update({'font.size': 16})
# bar_width = dpar.fid_bar
# bar_height = 2.0
# fig4 = plt.figure(figsize=(8,6), dpi=200)
# ax = fig4.add_subplot(1, 1, 1)
# L_mieplt = plt.imshow(10000*ldp_agl_rotated[:, :, 0], norm=colors.PowerNorm(gamma=1./4.), cmap=plt.cm.get_cmap('inferno'), extent=(-1*np.max(lx)/2, np.max(lx)/2, -1*np.max(ly)/2, np.max(ly)/2))
# plt.plot([0],[0], marker="*", color='y')
# cbar = fig4.colorbar(L_mieplt)
# cbar.ax.set_ylabel('F$_{disk}$ / F$_*$ x $10^5$')
# ts = ax.transData
# coords = [0, 0]
# tr = trf.Affine2D().rotate_deg_around(coords[0], coords[1], dpar.pphi)
# t = tr + ts
# rect = Rectangle((0 - bar_height/2, 0 - bar_width/2), bar_height, bar_width, color='k', alpha=1.0, transform=t)
# ax.add_patch(rect)
# plt.xlabel('$\\delta$ arcsec')
# plt.ylabel('$\\delta$ arcsec')
# plt.show()