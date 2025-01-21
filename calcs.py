import sys
import os

import numpy as np
import numba as nb

from disk_model_shaz import disk_model_fun
import ShazaliForwardModel as ShazMod

from mcmc_Shaz import lnlike


wl = np.loadtxt('Rodigas_export_updated.csv', skiprows=1,unpack=True, delimiter=',', usecols=(1))

amin=2.0
amax = 100
aexp= -3.7
mexp=29.0
f_asi=1.
f_ac=0.
f_wi=0.
f_th=0.
f_tr=0.
f_fe=0.

lnf = -0.01



m_asi = ShazMod.m_asi
m_ac = ShazMod.m_ac
m_wi = ShazMod.m_wi
m_th = ShazMod.m_th
m_tr = ShazMod.m_tr
m_fe = ShazMod.m_fe
pixsc = ShazMod.pixsc
npix = ShazMod.npix
zpix = ShazMod.zpix
sigma_up = ShazMod.sigma_up
sigma_y = ShazMod.sigma_y
sigma_d =  ShazMod.sigma_d
phang = ShazMod.phang
scatt_type =  ShazMod.scatt_type
projected_separation_pts_au = ShazMod.projected_separation_pts_au

param_list = amin, aexp, mexp, f_asi, f_ac, f_wi, f_th, f_tr, f_fe, lnf


ld, ldp, lx, ly, fluxmodel = disk_model_fun(wl, amin, amax, aexp, mexp, f_asi, f_ac, f_wi, f_th, f_tr, f_fe,
                                            m_asi, m_ac, m_wi, m_th, m_tr, m_fe, pixsc, npix, zpix, sigma_up, 
                                            sigma_y, sigma_d, phang, scatt_type, projected_separation_pts_au)

wl, fl, err = np.loadtxt('Rodigas_export_updated.csv', skiprows=1,unpack=True, delimiter=',', usecols=(1,2,6))
fluxmeasured = fl
fluxerr = err
projected_separation_pts_au = np.array([75.])

def lnprior(param_list):
    amin, aexp, mexp, f_asi, f_ac, f_wi, f_th, f_tr, f_fe, lnf = param_list
    if 0.01 < amin < 4. and -5. < aexp < -1. and 27. < mexp < 33. and 0. <= f_asi <= 1. and 0. <= f_ac <= 1. \
            and 0. <= f_wi <= 1. and 0. <= f_th <= 1. and 0. <= f_tr <= 1. and 0. <= f_fe <= 1. \
            and 0.99 < f_asi + f_ac + f_wi + f_th + f_tr + f_fe < 1.01 and -0.1 < lnf < 0.1:
        return 0.0
    return -np.inf

#lnlike imported from mcmc
    

def lnprob(param_list):
    lp = lnprior(param_list)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(param_list)

print('the results of lnprior', lnprior(param_list))
print('the results of lnlike', lnlike(param_list))
print('the results of lnprob', lnprob(param_list))
