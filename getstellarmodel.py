import numpy as np
from getkurucz import getkurucz_fun
# get a single stellar model at a given Teff and (if wanted) logg
# uses code from fit_star_kurucz6.pro and the calls getkurucz.pro
# From Alycia's IDL code getstellarmodel


def getstellarmodel_fun(Teff, **kwargs):
    # output: starmodel, w
    # input: Teff(from radpressure), logg(from radpressure), modeldir(from radpressure)

    logg = kwargs.pop('logg', 5.0)  # Set logg to 5.0 is not set by user
    stardir_kurucz = kwargs.pop('stardir_kurucz', './stellarmodels/')  # Set directory if not set by user
    mfile, num, label, teff_kurucz, label, grav = np.loadtxt(stardir_kurucz+'list_models',
                                                             dtype='object,i,object,f,object,f', unpack=True)
    Tminloc = np.argmin(abs(teff_kurucz - Teff))
    Tmin = teff_kurucz[Tminloc]
    gminloc = np.argmin(np.abs(grav - logg))
    gmin = grav[gminloc]
    uset = np.argwhere(np.logical_and(np.equal(teff_kurucz, Tmin), np.equal(grav, gmin)))
    print(Tmin, gmin, uset)
    starmodel, wlfull = getkurucz_fun(uset, mfile)
    starmodel = starmodel.flatten()
    wlfull = wlfull.flatten()
    return starmodel, wlfull

