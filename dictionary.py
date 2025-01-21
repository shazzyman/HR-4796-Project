"""
This Python script creates a dictionary the refractive indices and densities for various dust grain compositions
commonly found in astrophysical debris disks. The main features include:

1. **Wavelength Range and Parameters**:
   - Defines the wavelength range (`wlo`, `whi`) and grain size parameters (`amax`, `aexp`) for calculations.
   - Supports customization for specific disks by adjusting the wavelength range and number of wavelengths.

2. **Bruggeman's Mixing Rule**:
   - Implements the `bruggeman2` function to compute the effective refractive index for porous materials.
   - Combines two components with specified refractive indices to simulate porous grain properties.

3. **Chemical Composition Dictionary**:
   - Provides a dictionary (`nk_dict`) linking chemical types (e.g., Astrosilicate, Amorphous Carbon, Water Ice) to their 
     respective refractive index files.
   - Supports additional compositions like Tholins, Troilite, and Iron.

4. **Refractive Index Calculation**:
   - The `calc_m_val` function computes the complex refractive index (`n - i*k`) for a given chemical type.
   - Supports porosity calculations by applying Bruggeman's rule to adjust refractive indices.

5. **Precomputed Values for Common Compositions**:
   - Computes and stores refractive indices for standard compositions like:
     - Astrosilicate (`m_val_asi`)
     - Amorphous Carbon (`m_val_ac`)
     - Water Ice (`m_val_wi`)
     - Tholins (`m_val_th`)
     - Troilite (`m_val_tr`)
     - Iron (`m_val_fe`)

6. **Chemical Properties Dictionary**:
   - A dictionary (`chem_dict`) associates chemical types with their refractive indices, file paths, and densities.
   - Enables quick access to predefined material properties for modeling and simulations.

7. **Flexibility and Extendibility**:
   - Modular design allows for easy addition of new materials or scattering types.
   - The `calc_m_val` function is compatible with both Mie and porous scattering models, making it adaptable for 
     diverse astrophysical applications.
"""


import numpy as np
from scipy import interpolate
from scipy.ndimage.interpolation import rotate
from scipy import optimize
import disk_parameters as dpar
from collections import defaultdict

# Set wavelength range and maximum grain size   *Edit for specific disk*
wlo = 0.5737    # Minimum wavelength
whi = 3.7565    # Maximum wavelength
nwl = 11       # Number of wavelengths
wl = np.linspace(wlo, whi, num=nwl)
amax = 100.
aexp = -3.0

def bruggeman2(eff, *args):
    effc = eff[0:eff.size//2] + 1j*eff[eff.size//2:]
    e1=args[0]+1j*args[1]
    e2=(1.0 + np.zeros(e1.size)) + 1j*(0.0 + np.zeros(e1.size))
    minc = 0.25 * (e1 - effc) / (e1 + 2 * effc) + 0.75 * (e2 - effc) / (e2 + 2 * effc)
    return np.concatenate((np.real(minc), np.imag(minc)))

#Nk Files
nk_dict = {0: './optical_constants/Astro_silicate_Draine_2003.txt', #Astrosilicate
           1: './optical_constants/amorphousC_ACAR_Zubko.txt', #Amorphous Carbon
           2: './optical_constants/waterice_Henning.txt', #Water Ice
           3: './optical_constants/tholins.txt', #Tholins
           4: './optical_constants/troilite.txt', #Troilite
           5: './optical_constants/iron_Henning.txt', #Iron
           6: './optical_constants/brug_astrosil80_waterice_por30.txt', #Astro/WI density = 1.99
           7: './optical_constants/brug_olivineIP80_orthopyr_por0.txt', #Orthopyroxene density = ?
           8: './brug_olivineIP80_orthopyr_por0.txt'#Iron Poor Ortho density = ?
           }
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

#Load dust refractive indices
porosity = False #True if using porous spheres
scatt_type = dpar.scatt_type 
n0 = 1.0 +np.zeros(wl.size)
k0 = 0.1 + np.zeros(wl.size)
m_guess = np.concatenate((n0, k0)) #Used if 'Porous' chosen
m_brug = np.zeros(m_guess.shape)

def calc_m_val(chem_type):
    if chem_type == ('Asi' or 'asi'):
        nk_file_asi = nk_dict[0]
        nkdat_asi = np.loadtxt(nk_file_asi, skiprows=5, unpack=True)
        wlnk_asi = np.flipud(nkdat_asi[0, :])
        nf_asi = np.flipud(1 + nkdat_asi[3, :])
        kf_asi = np.flipud(nkdat_asi[4, :])
        n_asi = np.interp(wl, wlnk_asi, nf_asi)
        k_asi = np.interp(wl, wlnk_asi, kf_asi)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_asi,k_asi))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_asi - 1j*k_asi
            
    elif chem_type == ('A/WI' or 'a/wi'):
        nk_file_asi = nk_dict[6]
        nkdat_asi = np.loadtxt(nk_file_asi, unpack=True)
        wlnk_asi = np.flipud(nkdat_asi[0, :])
        nf_asi = np.flipud(1 + nkdat_asi[1, :])
        kf_asi = np.flipud(nkdat_asi[2, :])
        n_asi = np.interp(wl, wlnk_asi, nf_asi)
        k_asi = np.interp(wl, wlnk_asi, kf_asi)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_asi,k_asi))
            m_asi = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_asi = n_asi - 1j*k_asi
    
    elif chem_type == ('Orth'or 'orth'):
        nk_file_asi = nk_dict[7]
        nkdat_asi = np.loadtxt(nk_file_asi, unpack=True)
        wlnk_asi = np.flipud(nkdat_asi[0, :])
        nf_asi = np.flipud(1 + nkdat_asi[1, :])
        kf_asi = np.flipud(nkdat_asi[2, :])
        n_asi = np.interp(wl, wlnk_asi, nf_asi)
        k_asi = np.interp(wl, wlnk_asi, kf_asi)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_asi,k_asi))
            m_asi = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_asi = n_asi - 1j*k_asi

    elif chem_type == ('Ac' or 'ac'):
        nk_file_ac = nk_dict[1]
        wlnk_ac, nf_ac, kf_ac = np.loadtxt(nk_file_ac, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_ac = np.interp(wl, wlnk_ac, nf_ac)
        k_ac = np.interp(wl, wlnk_ac, kf_ac)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_ac,k_ac))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_ac - 1j*k_ac
            
    elif chem_type == ('WI' or 'wi'):
        nk_file_wi = nk_dict[2]
        wlnk_wi, nf_wi, kf_wi = np.loadtxt(nk_file_wi, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_wi = np.interp(wl, wlnk_wi, nf_wi)
        k_wi = np.interp(wl, wlnk_wi, kf_wi)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_wi,k_wi))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_wi - 1j*k_wi
        
    elif chem_type == ('Th' or 'th'):
        # Tholins
        nk_file_th = nk_dict[3]
        wlnk_th, nf_th, kf_th = np.loadtxt(nk_file_th, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_th = np.interp(wl, wlnk_th, nf_th)
        k_th = np.interp(wl, wlnk_th, kf_th)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_th,k_th))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_th - 1j*k_th
            
    elif chem_type == ('Tr' or 'tr'):
        nk_file_tr = './optical_constants/troilite.txt'
        wlnk_tr, nf_tr, kf_tr = np.loadtxt(nk_file_tr, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_tr = np.interp(wl, wlnk_tr, nf_tr)
        k_tr = np.interp(wl, wlnk_tr, kf_tr)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_tr,k_tr))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_tr - 1j*k_tr
    
    elif chem_type == ('Fe' or 'fe'):
        nk_file_fe = './optical_constants/iron_Henning.txt'
        wlnk_fe, nf_fe, kf_fe = np.loadtxt(nk_file_fe, skiprows=2, usecols=(0, 1, 2), unpack=True)
        n_fe = np.interp(wl, wlnk_fe, nf_fe)
        k_fe = np.interp(wl, wlnk_fe, kf_fe)
        if porosity:
            m_brug = optimize.root(bruggeman2, x0=m_guess, args=(n_fe,k_fe))
            m_val = m_brug.x[0:m_brug.x.size//2] - 1j*(m_brug.x[m_brug.x.size//2:])
        else:
            m_val = n_fe - 1j*k_fe 
    
    else:
        raise ValueError('Chemical type must be "Astrosilicate", "Amorphous Carbon", or "Water Ice"')
        
        
    
    
    return m_val

m_val_asi = calc_m_val('Asi')
m_val_ac = calc_m_val('Ac')
m_val_wi = calc_m_val('WI')
m_val_th = calc_m_val('Th')
m_val_tr = calc_m_val('Tr')
m_val_fe = calc_m_val('Fe')


    
chem_dict = {'Asi' or 'asi': ('./optical_constants/Astro_silicate_Draine_2003.txt', m_val_asi, 3.3),#Astrosilicate
            'Ac' or 'ac': ('./optical_constants/amorphousC_ACAR_Zubko.txt',  m_val_ac, 2.2),  #Amorphous Carbon
            'Wi' or 'wi': ('./optical_constants/waterice_Henning.txt', m_val_wi, 1.),  #Water Ice
            'Th' or 'th': ('./optical_constants/tholins.txt', m_val_th, 1.5), #Tholins
            'Tr' or 'tr': ('./optical_constants/troilite.txt', m_val_tr, 4.61), #Troilite
            'Fe' or 'fe': ('./optical_constants/iron_Henning.txt', m_val_fe, 7.87), #Iron
            'A/W' or 'a/w': ('./optical_constants/brug_astrosil80_waterice_por30.txt'), #Astro/WI density = 1.99
            'Orth' or 'orth': ('./optical_constants/brug_olivineIP80_orthopyr_por0.txt'), #Orthopyroxene density = ?
            'IPOr' or 'ipor': ('./brug_olivineIP80_orthopyr_por0.txt')  #Iron Poor Ortho density 
            }




