import numpy as np
# From Alycia's IDL code getstellarmodel
# Kurucz models all have 1221 elements and go to 160 mu
# Each file has header line, then lines with 3 columns:
# 1) wavelengths (nanometers)
# 2) surface brightnesses per unit frequency, with absorption lines
#    (erg s^-1 cm^-2 Hz^-1 steradian^-1)
# 3) surface brightnesses per unit frequency, intrinsic continuum only
#    (erg s^-1 cm^-2 Hz^-1 steradian^-1)


def getkurucz_fun(uset, mfile):
    # output: starmodel, smoothw
    # input: uset (from getstellarmodel.pro), file (from getstellarmodel.pro)

    kdir = './stellarmodels/'
    nteff = uset.size
    starmodel = np.zeros((nteff, 2212))
    smoothw = np.zeros((nteff, 2212))

    for icnt in range(0, nteff):
        print('reading Kurucz file ', mfile[uset[icnt]])
        tmp_str1 = np.str(mfile[np.int(uset[icnt])]).split(':')
        w, fnu = np.loadtxt(kdir+tmp_str1[0], dtype='f, d', unpack=True, skiprows=1)
        w = w / 1.e3                # convert nm to mu
        fnu = fnu*1.e23             # 1.d4/1.d7/1.d-26      # convert erg/s/cm^2/Hz/ster to Jy/ster
        # extrapolate to 1000 mu with a straight line starting at 10 mu (where
        # model starts to be sparsely sampled)
        a = (w >= 10.).nonzero()
        b = (w < 10.).nonzero()
        r = np.polyfit(np.log10(w[a]), np.log10(fnu[a]), 1)
        tmp1 = np.min(w[a]) + np.arange(1000.)*0.991
        newfnu = 10.**(r[1] + r[0]*np.log10(tmp1))
        smoothw[icnt, :] = np.concatenate((w[b], tmp1))
        starmodel[icnt, :] = np.concatenate((fnu[b], newfnu))

    return starmodel, smoothw
