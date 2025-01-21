import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from photutils.aperture import EllipticalAnnulus
from photutils.aperture import RectangularAperture
from photutils.aperture import aperture_photometry
from ShazaliForwardModel import data
from ShazaliForwardModel import L_mieplt
import disk_parameters as dpar
import ShazaliForwardModel as ShazMod

pixsc = 1. / (dpar.px_size * dpar.dsun)

def Elliptical_Annulus_sum(spec_wl_data, theta, center_x, center_y, a_in, a_out, b_in, b_out):
    position = (center_x, center_y)
    
    aperture1 = EllipticalAnnulus(position, a_in, a_out, b_out, b_in, theta = theta )
    phot1_table = aperture_photometry(spec_wl_data, aperture1)
    phot1_table['aperture_sum'].info.format = '%.8g' 
    print(phot1_table)
    
    print(phot1_table['aperture_sum'])
    
    return(aperture1, phot1_table)

    
spec_wl_data = data[76]
theta = np.pi / 2 - dpar.pphi * np.pi / 180
center_x = 82.5
center_y = 82.5
#a_in = dpar.rmin * pixsc 
#a_out = dpar.rmax * pixsc
#b_out = (np.sqrt((dpar.rmax) ** 2)*(np.cos((dpar.theta_i * np.pi / 180)) ** 2))  * pixsc
#b_in = (np.sqrt((dpar.rmin) ** 2)*(np.cos((dpar.theta_i) * np.pi / 180) ** 2)) * pixsc
a_in = 66 * pixsc 
a_out = 120 * pixsc
b_out = ((120) * np.cos((dpar.theta_i * np.pi / 180)))  * pixsc
b_in = ((66) * np.cos((dpar.theta_i * np.pi / 180)))  * pixsc


aperture1, phot1_table = Elliptical_Annulus_sum(spec_wl_data, theta, center_x, center_y, a_in, a_out, b_in, b_out)

# aperture2 = RectangularAperture(position, r = (dpar.rmin * pixsc))
# phot2_table = aperture_photometry(spec_wl_data, aperture2)
# phot2_table['aperture_sum'].info.format = '%.8g' 
# print(phot2_table)



# =============================================================================
#       - phot2_table['aperture_sum'])
# =============================================================================

# =============================================================================
# def bounding_ellipses(self):
#         """
#         Compute the semimajor axis of the two ellipses that bound the
#         annulus where integrations take place.
# 
#         Returns
#         -------
#         sma1, sma2 : float
#             The smaller and larger values of semimajor axis length that
#             define the annulus bounding ellipses.
#         """
#         if self.linear_growth:
#             a1 = self.sma - self.astep / 2.
#             a2 = self.sma + self.astep / 2.
#         else:
#             a1 = self.sma * (1. - self.astep / 2.)
#             a2 = self.sma * (1. + self.astep / 2.)
# 
#         return a1, a2
# 
# bounding_ellipses(L_mieplt)
# =============================================================================

ldp_agl_rotated = ShazMod.ldp_agl_rotated
colors = ShazMod.colors
lx = ShazMod.lx
ly = ShazMod.ly

fig4 = plt.figure(figsize=(8,6), dpi=200)
ax = fig4.add_subplot(1, 1, 1)
x = ldp_agl_rotated[:, :, 76]
L_mieplt = ax.imshow(10000*ldp_agl_rotated[:, :, 76], norm=colors.PowerNorm(gamma=1./4.), cmap=plt.cm.get_cmap('inferno'),)
#L_mieplt = ax.pcolormesh(lx,ly,10000*ldp_agl_rotated[:, :, 76], norm=colors.PowerNorm(gamma=1./4.), cmap=plt.cm.get_cmap('inferno'), extent=(-1*np.max(lx)/2, np.max(lx)/2, -1*np.max(ly)/2, np.max(ly)/2))
ax.plot([0],[0], marker="*", color='y')
cbar = fig4.colorbar(L_mieplt)
cbar.ax.set_ylabel('F$_{disk}$ / F$_*$ x $10^5$')
coords = [0, 0]
aperture1.plot(axes=ax, color='white', lw=1.5, alpha=0.5)
plt.show()
# aperture2.plot(color='blue', lw=1.5, alpha=0.5)
# 
# =============================================================================
