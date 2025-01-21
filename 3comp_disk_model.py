#!/usr/bin/env python

"""
This Python script models the forward calculation of flux from debris disks around stars by leveraging
advanced computational techniques and scientific principles. It incorporates multiple functions to simulate 
dust properties, scattering phase functions, flux calculations, and 3D to 2D disk projections. The key 
features and components include:

1. **Dust Size and Distribution**:
   - Generates a log-spaced array of dust grain sizes and calculates their normalized size distribution 
     based on a power-law exponent.

2. **Scattering Phase Functions**:
   - Uses lookup tables to compute scattering phase functions for different dust grain sizes and 
     compositions.
   - Supports Mie, Porous, and Agglomerate scattering models with appropriate refractive index 
     tables.

3. **Flux Calculation**:
   - Computes the luminosity of the disk by considering dust properties, scattering phase functions, 
     and distances from the star.
   - Implements an efficient voxel-based approach using numba for speed optimization.

4. **3D to 2D Disk Projection**:
   - Projects 3D disk properties into a 2D image using positional and flux information.
   - Averages the projected flux over spatial regions to match observational constraints.

5. **Forward Model Integration**:
   - Combines all calculations to produce synthetic disk flux profiles for various dust parameters.
   - Supports integration with external observational data for model fitting.

6. **Optimization and Performance**:
   - Accelerates computations using numba's JIT compilation for critical functions.
   - Utilizes HDF5 files for efficient I/O operations on large lookup tables.
"""
# Imports
import h5py
import numpy as np
from astropy.nddata.blocks import block_reduce
from scipy import interpolate
import numba as nb
# User defined
import disk_parameters as dpar


# Create array of dust sizes
def dust_sizes(amin, amax, aexp):
    # Dust size range

    na = 10         # number of size bins
    a = np.geomspace(amin, amax, num=na)  # log-spaced array of particle sizes
    abin = (a**aexp)/np.sum(a**aexp)   # relative number of particles of each size (normalized so the total is 1)
    return a, abin


# Get phase functions for each grain size and wavelength
@nb.jit(nb.types.Tuple((nb.float64[:,:], nb.float64[:, :, :]))(nb.int64, nb.int64, nb.float64[:], nb.float64[:],
                                nb.complex128[:], nb.complex128[:], nb.complex128[:], 
                                nb.float64, nb.float64, nb.float64, 
                                nb.float64[:], nb.complex128[:], nb.float64[:], nb.float64[:,:],
                                nb.float64[:], nb.complex128[:], nb.float64[:], nb.float64[:,:]), nopython=True)
def get_phase_new(nwl, na, a, wl, m_1, m_2, m_3, f_1, f_2, f_3, x_arr, m_arr,
              qsca_arr, ph_arr, hk_x_arr, hk_m_arr, hk_qsca_arr, hk_ph_arr):
    qsca_tab = np.zeros((na, nwl))
    ph_tab = np.zeros((na, nwl, 60))
    for wli in range(0, nwl):
        for ai in range(0, na):
            # find scattering parameters from tables
            x = 2*np.pi*a[ai]/wl[wli]
            xtab_index = np.argmin(np.abs(x_arr - x))
            hk_xtab_index = np.argmin(np.abs(hk_x_arr - x))
            mtab_index_1 = xtab_index + np.argmin(np.abs(m_arr - m_1[wli]))
            mtab_index_2 = xtab_index + np.argmin(np.abs(m_arr - m_2[wli]))
            mtab_index_3 = xtab_index + np.argmin(np.abs(m_arr - m_3[wli]))
            qsca_tab[ai, wli] = f_1 * qsca_arr[mtab_index_1] + f_2 * qsca_arr[mtab_index_2] + \
                                f_3 * qsca_arr[mtab_index_3] 
            ph_num = (a[ai] ** 2) * (f_1*qsca_arr[mtab_index_1]*ph_arr[mtab_index_1] + f_2*qsca_arr[mtab_index_2]*ph_arr[mtab_index_2] +
                        f_3*qsca_arr[mtab_index_3]*ph_arr[mtab_index_3]) 
            ph_denom = (a[ai] ** 2) * (f_1 * qsca_arr[mtab_index_1] + f_2 * qsca_arr[mtab_index_2] +
                        f_3 * qsca_arr[mtab_index_3])
            ph_tab[ai, wli, :] = ph_num/ph_denom

    return qsca_tab, ph_tab


# Calculate the flux from each voxel
@nb.jit((nb.float64[:, :, :, :])(nb.int64, nb.int64, nb.float64[:, :],
        nb.float64[:, :], nb.int64, nb.float64[:, :, :], nb.float64[:, :, :],
        nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :]), nopython=True)
def calc_flux(npix, zpix, a_tiled, abin_tiled, nwl, sigma_num, sigma_d, pg, phang, qsca):
    # Calculate luminosity with Mie or agglomerate phase fn's
    ld = np.zeros((npix, npix, zpix, nwl))
    for xi in range(0, npix):
        for yi in range(0, npix):
            for zi in range(0, zpix):
                pgi = pg[:, :, np.int(np.round((180 / np.pi) * phang[xi, yi, zi]))]
                ld[xi, yi, zi, :] = sigma_num[xi, yi, zi] * np.sum(abin_tiled * pgi[:, :] * qsca *
                                            ((a_tiled * 1e-6) ** 2), axis=0)/((2 * sigma_d[xi, yi, zi] + 1e-10) ** 2)
    return ld


# Project the 3D calculations into 2D
@nb.jit((nb.float64[:, :, :])(nb.int64, nb.int64, nb.int64, nb.float64[:, :, :],
        nb.float64[:, :, :, :]), nopython=True)
def project_disk(npix, zpix, nwl, sigma_y, ld):
    # Project disk
    # Figure out where each pixel ends up in 2D image, add flux from each
    ldp = np.zeros((npix, npix, nwl))
    for xi in range(0, npix):
        for zi in range(0, zpix):
            yind1 = np.floor(sigma_y[xi, :, zi])
            yamt1 = 1. - (sigma_y[xi, :, zi] - np.floor(sigma_y[xi, :, zi]))
            yind2 = np.ceil(sigma_y[xi, :, zi])
            yamt2 = sigma_y[xi, :, zi] - np.floor(sigma_y[xi, :, zi])
            for yi in range(0, npix):
                yi1 = np.int64(yind1[yi])
                yi2 = np.int64(yind2[yi])
                ldp[xi, yi1, :] = ldp[xi, yi1, :] + yamt1[yi]*ld[xi, yi, zi, :]
                ldp[xi, yi2, :] = ldp[xi, yi2, :] + yamt2[yi]*ld[xi, yi, zi, :]
    return ldp


# Main body of the forward model for the flux
def disk_model_fun_new(wl, amin, amax, aexp, mexp, f_1, f_2, f_3, m_1, m_2, m_3, 
                   pixsc, npix, zpix, sigma_up, sigma_y, sigma_d, phang, scatt_type, projected_separation_pts_au):
    # Calculate Qabs, Qsca, Pg for each particle size, wavelength
    #    For now assume each part of the disk has the same size distribution
    #    and average over that distribution.

    thet = (np.pi/180.)*(np.linspace(0, 360, num=361))
    thet_tab = (np.pi/180.)*(np.linspace(0, 180, num=60))

    a, abin = dust_sizes(amin, amax, aexp)
    nwl = wl.size
    na = a.size

    pg = np.zeros((na, nwl, 361))

    # Choose lookup table file and porosity based on chosen scattering type and refractive index.
        # high_k files are used for metallic components (k > 1) where doing agglomerate
        # calculations becomes slow.
    if scatt_type == ('Mie' or 'mie'):
        table_file = 'MieTableMaxR100nwl10thres3deg.hdf5'
        high_k_table_file = 'MieTableMetalMaxR100nwl10thres3deg.hdf5'
        phi_pf = 1.
        phi_mpf = 1.
    elif scatt_type == ('Porous' or 'porous'):
        table_file = 'MieTableMaxR100nwl10thres3deg.hdf5'
        high_k_table_file = 'MieTableMetalMaxR100nwl10thres3deg.hdf5'
        phi_pf = 0.25
        phi_mpf = 0.25
    elif scatt_type == ('Agglomerate' or 'agglomerate'):
        table_file = 'AglTableMaxR100nwl10thres3deg.hdf5'
        high_k_table_file = 'MieTableMetalMaxR100nwl10thres3deg.hdf5'
        phi_pf = 0.25
        phi_mpf = 2.0
    else:
        raise ValueError('Scattering type must be "Mie", "Porous", or "Agglomerate".')

    table_vals = h5py.File(table_file, 'r', libver='latest', swmr=True)
    x_arr = np.array(table_vals['x'][:])
    m_arr = np.array(table_vals['m'][:])
    qsca_arr = np.array(table_vals['qsca'][:])
    ph_arr = np.array(table_vals['ph'][:, :])
    table_vals.close()

    high_k_table_vals = h5py.File(high_k_table_file, 'r', libver='latest', swmr=True)
    hk_x_arr = np.array(high_k_table_vals['x'][:])
    hk_m_arr = np.array(high_k_table_vals['m'][:])
    hk_qsca_arr = np.array(high_k_table_vals['qsca'][:])
    hk_ph_arr = np.array(high_k_table_vals['ph'][:, :])
    high_k_table_vals.close()

    qsca, pg_tab = get_phase_new(nwl, na, a, wl, m_1, m_2, m_3, f_1, f_2, f_3, 
                         x_arr, m_arr, qsca_arr, ph_arr, hk_x_arr, hk_m_arr, hk_qsca_arr, hk_ph_arr)
    for wli in range(0, nwl):
        for ai in range(0, na):
            y_spline = interpolate.splrep(thet_tab, pg_tab[ai, wli], k=1, t=None)
            pg_interp = interpolate.splev(thet[0:181], y_spline, der=0)
            pg_interp_full = np.concatenate((pg_interp, np.flip(pg_interp[:-1], 0)), axis=0)
            pg[ai, wli, :] = pg_interp_full     # /(2*np.pi*np.trapz(pg_interp_full[0:181]*sinthet[0:181], x=thet[0:181]))

    # Calculate luminosity with Mie and agglomerate phase fn's
    sigma_up = (10. ** mexp) * sigma_up / np.sum(sigma_up)
    sigma_num = sigma_up / np.trapz(
        (4. / 3.) * np.pi * (phi_pf * f_1 * 3.3 + phi_pf * f_2 * 2.2 + phi_pf * f_3 * 1.0) * ((a * 1e-4) ** 3) * abin, a)
    a_tiled = np.repeat(a[:, np.newaxis], nwl, axis=1)
    abin_tiled = np.repeat(abin[:, np.newaxis], nwl, axis=1)
    ld = calc_flux(npix, zpix, a_tiled, abin_tiled, nwl, sigma_num, sigma_d, pg, phang, qsca)
    del a_tiled
    del abin_tiled

    # Project disk
    # Figure out where each pixel ends up in 2D image, add flux from each
    ldp = project_disk(npix, zpix, nwl, sigma_y, ld)

    scale_image = 'no'
    if scale_image == 'yes':
        rbin_scale = pixsc
        ldp = (pixsc**2)*block_reduce(ldp, [rbin_scale, rbin_scale, 1], func=np.mean)

    lx = -1*dpar.px_size*(np.arange(npix) - npix/2)
    ly = -1*dpar.px_size*(np.arange(npix) - npix/2)

    # Extract the projected separations that are being fit and average over the number of pixels
        # in the spatial direction specified in the disk parameter file
    npspts = projected_separation_pts_au.size
    fluxmodel = np.empty((0,), float)
    for npspts_i in range(0, npspts):
        start_pts = np.int(npix / 2 + projected_separation_pts_au[npspts_i]/dpar.pxs - np.floor(dpar.nspx))
        stop_pts = np.int(start_pts + dpar.nspx)
        fluxmodel = np.append(fluxmodel, np.sum(np.average(ldp[start_pts:stop_pts, :, :], axis=0), axis=0))

    return ld, ldp, lx, ly, fluxmodel


