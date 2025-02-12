# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from tqdm import tqdm
from joblib import Parallel, delayed

# %%

F353 = np.load('Planck_Foreground/F353.npy')
F545 = np.load('Planck_Foreground/F545.npy')
F857 = np.load('Planck_Foreground/F857.npy')

freq = np.loadtxt('freq')
# %%


def func(v, p1, p2):
    return v**p1/(np.exp(v/p2) - 1)


def vfunc_base(x, y, z):
    (p1, p2), _ = curve_fit(func, np.array([353, 545, 857])/1000.,
                            np.array([x, y, z]))
    fl = np.random.choice([-0.1, 0.1])  # fluctuation
    p1 = p1 + p1 * fl  # fluctuation

    return np.array([p1, p2])


def p1p2(i):
    f353 = F353[i].flatten()
    f545 = F545[i].flatten()
    f857 = F857[i].flatten()
    size = F353[i].shape[0]
    out = Parallel(n_jobs=30)(delayed(vfunc_base)(x, y, z)
                              for x, y, z in zip(f353, f545, f857))
    out = np.array(out)
    p1 = out[:, 0].reshape(size, size)
    p2 = out[:, 1].reshape(size, size)
    return p1, p2


foregrounds_sync = []
params_1 = []
params_2 = []

for i in tqdm(range(F353.shape[0])):
    p1, p2 = p1p2(i)
    f_freq = []
    for f in freq:
        f = f / 1000.
        map = f**p1/(np.exp(f/p2) - 1)
        f_freq.append(map)

    f_freq = np.array(f_freq)
    f_freq = np.transpose(f_freq, (2, 1, 0))
    foregrounds_sync.append(f_freq)

    params_1.append(p1)
    params_2.append(p2)

foregrounds_sync = np.array(foregrounds_sync)
params_1 = np.array(params_1)
params_2 = np.array(params_2)


Jy_to_uK = 32.6 / freq**2

foregrounds = foregrounds_sync * 10**6 * Jy_to_uK

np.save('Raw_data/Foregrounds_1024_perturbed', foregrounds)
np.save('Raw_data/p1', params_1)
np.save('Raw_data/p2', params_2)
# %%
