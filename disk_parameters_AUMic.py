import numpy as np


# AU Mic Disk parameters
rmin = 8.8          # [AU] Belt inner edge
rmax = 40.3         # [AU] Belt outer edge
rmaxhalo = 80      # [AU} Halo cutoff
xsigma = 2.82       # Surface density exponent
xsigmahalo = -1.5   # Halo surface density exponent
ophi = 177.7        # [deg] Opening angle 177.7
pphi = 128.41       # Position angle
theta_i = 89.5      # Disk inclinaiton angle
sin_i = np.sin(theta_i*np.pi/180)
cos_i = np.cos(theta_i*np.pi/180)
tan_i2 = np.tan(theta_i*np.pi/360)

# pixel size
px_size = 0.051  # Pixel size in arcseconds 0.063 JWST NIRCam, 0.051 HST STIS
fid_bar = 0.86   # Fiducial bar in arcseconds
dsun = 9.9  # Stellar distance in pc
pxs = 2*np.tan(np.pi*px_size/(2.*3600.*180.))*(dsun/(4.84814e-6))  # pixel size in AU
nspx = 5  # Number of pixls averaged in the spatial direction