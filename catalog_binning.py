# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic_dd
import os
from colossus.cosmology import cosmology
from astropy.cosmology import FlatwCDM, z_at_value
import astropy.units as units
from joblib import Parallel, delayed
from itertools import product

# %%

# cosmo_params = {'flat': True, 'H0': 67.77, 'Om0': 0.307115,
#                 'Ob0': 0.048206, 'sigma8': 0.8228, 'ns': 0.96,
#                 'de_model': 'lambda', 'w0': -1}

cosmos = FlatwCDM(name='myCosmos', H0=67.77, Om0=0.307115, Ob0=0.048206, w0=-1)

cosmo = cosmology.fromAstropy(cosmos, sigma8=0.8228, ns=0.96)
# %%

Mpc = 3.0856776e19  # km
Lsun = 3.839e26  # unit W
Jy = 1e-26  # W/m^2/Hz
GHz = 1e9  # Hz
c_light = 3e8  # m/s

Lbox = 1 * 10**3  # 1 Gpc/h
subbox = 300  # Mpc/h

rad_to_deg = 57.3

freq_rest = 115.27
dnu = 0.1

Npix = 1024
h = cosmo.Hz(z=0) / 100.

# %%
middle_z = 1.0
d_comv = cosmo.comovingDistance(z_max=1.0)  # Mpc/h

d_front = d_comv - subbox/2
d_back = d_comv + subbox/2

z_front = z_at_value(cosmos.comoving_distance, d_front/h * units.Mpc)
z_back = z_at_value(cosmos.comoving_distance, d_back/h * units.Mpc)

freq_front = freq_rest/(1 + z_front)
freq_back = freq_rest/(1 + z_back)

freq_range = np.arange(freq_front, freq_back, -dnu)
z_range = freq_rest / freq_range - 1
d_los_bins = cosmo.comovingDistance(z_max=z_range)  # Mpc/h
spatial_bins = np.linspace(0, subbox, num=(Npix+1), endpoint=True)

print(freq_range)

# %%
catalog = np.load('LCatalog/LCO_halo_CO10_8-1000.npy')

x = catalog[:, 1]
y = catalog[:, 2]
z = catalog[:, 3]
Mh = catalog[:, 0]  # M_sun/h

# %%
mass_lumi_relation = np.loadtxt('LCatalog/halo_mass_and_CO_lumi')
mass = mass_lumi_relation[:, 0]  # M_sun/h
lumi = mass_lumi_relation[:, 1]  # L_sun

relation = interp1d(mass, lumi, kind='linear',
                    fill_value='extrapolate')  # Lco ~ Mh


# %%

def slice_snapshot(i, j, k):
    idx = np.where((x >= subbox * i) & (x < subbox * (i + 1))
                   & (y >= subbox * j) & (y < subbox * (j + 1))
                   & (z >= subbox * k) & (z < subbox * (k + 1)))[0]

    x_re = x[idx] - i * subbox
    y_re = y[idx] - j * subbox
    z_re = z[idx] - k * subbox

    return x_re, y_re, z_re, Mh[idx]


def halo_intensity(Mh, coord):
    CO_lumi = relation(Mh)
    dtheta = np.arctan(subbox / 2 / d_comv) * 2 / Npix  # radian
    d_los = coord + d_front
    z = z_at_value(cosmos.comoving_distance, d_los/h * units.Mpc)
    Jy_to_uK = 32.6 / (115.27/(1 + z))**2
    its = 1/(dtheta)**2 * (1/dnu) * (CO_lumi /
                                     (4 * np.pi * d_los**2 * (1 + z)**2)) \
        * (1 / GHz) * (Lsun / (Mpc * 10**3 / h)**2) / Jy * Jy_to_uK
    return its


def arr_gen(i, j, k, los='z', inverse=False):
    print('---finding to data in one data cube---')
    x, y, z, halo_mass = slice_snapshot(i, j, k)
    if los == 'x':
        coord = x
        spatial = [y, z]
    elif los == 'y':
        coord = y
        spatial = [x, z]
    else:
        coord = z
        spatial = [x, y]

    if inverse is True:
        coord = subbox - coord

    #halo_its_func = np.vectorize(halo_intensity)
    print('---calculate intensity for data---')
    #intensity = halo_its_func(halo_mass, coord)
    intensity = Parallel(n_jobs=20)(delayed(halo_intensity)(hm, co)
                                    for hm, co in zip(halo_mass, coord))

    intensity = np.array(intensity)

    d_los = coord + d_front

    sample = [spatial[0], spatial[1], d_los]
    bins = [spatial_bins, spatial_bins, d_los_bins]

    print('---binning---')
    arr, _, _ = binned_statistic_dd(np.column_stack(sample),
                                    intensity, statistic='sum',
                                    bins=bins)
    arr = np.nan_to_num(arr)
    return arr

# %%


num = np.arange(3)
los = ['x', 'y', 'z']
inverse = [True, False]

CO_arr = []
count = 0
for i, j, k, l, inv in product(num, num, num, los, inverse):
    print(count)
    arr = arr_gen(i, j, k, l, inv)

    CO_arr.append(arr)
    count = count + 1

CO_arr = np.array(CO_arr)

# CO_arr = Parallel(n_jobs=30)(delayed(arr_gen)(i, j, k, l, inv)
#                              for i, j, k, l, inv in
#                              product((num, num, num, los, inverse)))

# CO_arr = np.array(CO_arr)

np.save('Raw_data/Raw_CO_map_1024_low.npy', CO_arr)
