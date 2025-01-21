import numpy as np


# HR 4796 Disk parameters
scatt_type = 'Agglomerate' # *Choices are 'Mie', 'Porous', 'Agglomerate'*
rmin = 67.95        # [AU] Belt inner edge
rmax = 81.95         # [AU] Belt outer edge
rmaxhalo = 100     # [AU} Halo cutoff
xsigma = 10       # Surface density exponent
xsigmahalo = -10.0   # Halo surface density exponent
ophi = 177.7       # [deg] Opening angle 177.7
pphi = 206.8       # Position angle = 206.4
theta_i = 75.9      # Disk inclinaiton angle = 180+75.9
sin_i = np.sin(theta_i*np.pi/180)
cos_i = np.cos(theta_i*np.pi/180)
tan_i2 = np.tan(theta_i*np.pi/360)


# pixel size
px_size = 0.03  # Pixel size in arcseconds 0.063 JWST NIRCam, 0.051 HST STIS
fid_bar = 0.86   # Fiducial bar in arcseconds
dsun = 70.77  # Stellar distance in pc
pxs = 2*np.tan(np.pi*px_size/(2.*3600.*180.))*(dsun/(4.84814e-6))  # pixel size in AU
nspx = 5  # Number of pixls averaged in the spatial direction


#Set flags for parameters, set the priors

amin = True
aexp = False
mexp = True
mll = True

comptypes = ['Asi', 'Ac', 'Wi']

comptypes_array = np.array(comptypes)

amin_prior = [2, 10]
aexp_prior = [-5, -1]
mexp_prior = [29, 32]
comptypes_prior = [0, 1]
mll_prior = [-1, 1]