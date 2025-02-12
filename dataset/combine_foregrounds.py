import numpy as np
from joblib import Parallel, delayed
import os
import healpy as hp
from itertools import product


dir = 'all_foregrounds'
ls = os.listdir(dir)
ls.sort()
ls = ls[::-1]

fg_maps = [os.path.join(dir, name) for name in ls]


area = 7.449 * 7.449
min_dec = 50.
min_ra = 0.
max_ra = 360.
resolution = 1024


angle = area ** 0.5
num = int((max_ra - min_ra) / angle)


def arr_gen(hpx, i, plus=1):
    longitude = np.linspace(angle * i + min_ra, angle * (i + 1)
                            + min_ra, resolution) * np.ones((resolution, 1))
    if plus == 1:
        latitude = np.linspace(min_dec + angle, min_dec,
                               resolution).reshape(-1, 1) * np.ones(resolution)
    elif plus == 0:
        latitude = np.linspace(-min_dec, -min_dec - angle,
                               resolution).reshape(-1, 1) * np.ones(resolution)
    else:
        print('Wrong number')

    return hp.get_interp_val(hpx, longitude, latitude, lonlat=True)


ii_p = list(zip(np.arange(num, dtype=np.int32),
            np.ones(num, dtype=np.int32)))
ii_m = list(zip(np.arange(num, dtype=np.int32),
            np.zeros(num, dtype=np.int32)))

ii = ii_p + ii_m

slice_dir = 'fg_slices'
os.makedirs(slice_dir, exist_ok=True)

foregrounds = []
for filename in fg_maps:
    hpx = np.load(filename)
    print(filenames)

    fg_slices = Parallel(n_jobs=20)(
        delayed(arr_gen)(hpx, i, f) for i, f in ii)

    fg_slices = np.array(fg_slices)

    filename = os.path.join(slice_dir, 'arr_' + filename.split('/')[-1][3:-4])

    np.save(filename, fg_slices)

    foregrounds.append(fg_slices)

foregrounds = np.array(foregrounds)
# freq, num, dim, dim

foregrounds = np.transpose(foregrounds, (1, 2, 3, 0))

np.save('Raw_data/Foregrounds_all_1024_high_res.npy', foregrounds)
