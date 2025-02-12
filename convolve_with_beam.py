import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.interpolate import interp1d
from itertools import product
from scipy.signal import convolve2d as conv2d
from skimage.transform import downscale_local_mean
from joblib import Parallel, delayed
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='signal',
                    help='signal or foreground')

args = parser.parse_args()
t = args.type

print(t)
sys.stdout.flush()
sys.stderr.flush()


def beamfunc(max_angle, freq):
    angle = np.arange(max_angle/1000., max_angle,
                      max_angle/1000.) * np.pi / 180.
    value = np.zeros_like(angle)
    wavelength = 3 * 10**8 / (freq * 10**9)
    D = 10.4
    u = np.sin(angle) / (wavelength / D / np.pi)
    idx = np.abs(u < 7.0156)
    value[idx] = (2 * jv(1, u[idx]) / u[idx])**2
    return angle, value


max_angle = 0.1
freq = np.loadtxt('freq')

dtheta = 0.0001270324944207476 / 4.  # Npix = 1024

half_pixels = 42


lines = []
for f in freq:
    a, b = beamfunc(max_angle, f)

    line = interp1d(a, b, kind='cubic', fill_value=0, bounds_error=False)

    lines.append(line)

kernel = np.zeros(shape=(half_pixels * 2,
                  half_pixels * 2, freq.shape[0]))
for k in range(len(lines)):
    for i, j in product(range(84), range(84)):
        distance = np.sqrt((i - 42)**2 + (j - 42)**2)
        angle = dtheta * distance
        val = lines[k](angle)
        kernel[i, j, k] = val

kernel = downscale_local_mean(kernel, factors=(4, 4, 1))

kernel = kernel / (np.sum(kernel, axis=(0, 1)))

print(kernel.shape)

assert kernel.shape[-1] == 51

np.save('kernel', kernel)

def beam_convolve(image, kernel, out_dims):

    conv_img = np.zeros_like(image)
    for i in range(conv_img.shape[-1]):
        conv_img[:, :, i] = conv2d(
            image[:, :, i], kernel[:, :, i], mode='same')

    assert image.shape == conv_img.shape

    ratio = image.shape[0] // out_dims

    pool = downscale_local_mean(conv_img, factors=(ratio, ratio, 1))
    return pool

# Testing:

os.makedirs('Exp', exist_ok=True)
os.makedirs('Data', exist_ok=True)

if t == 'signal':

    raw_co_map = np.load('Raw_data/Raw_CO_map_1024_low.npy')

    beamed_img_example = beam_convolve(raw_co_map[0], kernel, 128)

    np.save('Exp/beamed_map_example_low', beamed_img_example)

    gen_list = Parallel(n_jobs=10)(delayed(beam_convolve)(img, kernel, 128)
                                   for img in raw_co_map)

    beam_maps = np.array(gen_list)
    print(beam_maps.shape)

    np.save('Data/beamed_CO_signal_low', beam_maps)

    container = np.zeros(
        shape=(raw_co_map.shape[0], 128, 128, raw_co_map.shape[-1]))

    ratio = raw_co_map.shape[1] // 128

    for i in range(raw_co_map.shape[0]):
        container[i] = downscale_local_mean(
            raw_co_map[i], factors=(ratio, ratio, 1))

    np.save('Data/CO_signal_low', container)

elif t == 'foreground':

    raw_f_map = np.load('Raw_data/Foregrounds_all_1024_high_res_laf_lof.npy')
    print(raw_f_map.shape)
    beamed_f_example = beam_convolve(raw_f_map[0], kernel, 128)
    np.save('Exp/beamed_f_example_all_high_res', beamed_f_example)

    gen_list = Parallel(n_jobs=20)(delayed(beam_convolve)(img, kernel, 128)
                                   for img in raw_f_map)
    beam_maps = np.array(gen_list)
    print(beam_maps.shape)
    np.save('Data/beamed_foregrounds_all_high_res_laf_lof', beam_maps)

    container = np.zeros(
        shape=(raw_f_map.shape[0], 128, 128, raw_f_map.shape[-1])
    )

    ratio = raw_f_map.shape[1] // 128

    for i in range(raw_f_map.shape[0]):
        container[i] = downscale_local_mean(
            raw_f_map[i], factors=(ratio, ratio, 1)
        )

    np.save('Data/foregrounds_all_high_res_laf_lof', container)

else:
    print('Error args')

