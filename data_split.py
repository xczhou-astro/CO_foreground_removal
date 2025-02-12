# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--low', action='store_true',
                    help='low co or not')

args = parser.parse_args()
low_flag = args.low

if low_flag is False:
    # %%
    CO = np.load('Data/CO_signal.npy')
    beamed_fore = np.load('Data/beamed_foregrounds_all_high_res_laf_lof.npy')
    beamed_CO = np.load('Data/beamed_CO_signal.npy')

else:
    CO = np.load('Data/CO_signal_low.npy')
    beamed_fore = np.load('Data/beamed_foregrounds_all_high_res_laf_lof.npy')
    beamed_CO = np.load('Data/beamed_CO_signal_low.npy')


# %%
def split(signal, b_fg, b_sig, signal_idx, fg_idx, base_dir, k=0, flag='Train'):
    co_signal = signal[signal_idx]
    b_signal = b_sig[signal_idx]

    mixed = b_signal + b_fg[fg_idx]
    
    saved_filename = base_dir + '/' + flag + '/arr_{:04d}.npz'.format(k)

    np.savez_compressed(saved_filename, src=mixed, tar=co_signal)
    # np.savez_compressed(saved_filename, src=mixed, tar=b_fg[fg_idx])


base_dir = 'Dataset/Data_corr_split'
if low_flag is True:
    base_dir += '_low'


os.makedirs(base_dir + '/Train', exist_ok=True)
os.makedirs(base_dir + '/Test', exist_ok=True)

np.random.seed(1234)
selected_CO_index_train = np.random.choice(
    np.arange(CO.shape[0]), 130, replace=False)
np.random.seed(4321)
selected_Fore_index_train = np.random.choice(
    np.arange(beamed_fore.shape[0]), 70, replace=False)

selected_CO_index_test = np.setdiff1d(
    np.arange(CO.shape[0]), selected_CO_index_train)
selected_Fore_index_test = np.setdiff1d(
    np.arange(beamed_fore.shape[0]), selected_Fore_index_train)

Parallel(n_jobs=10)(delayed(split)(
    CO, beamed_fore, beamed_CO, i[0], i[1], base_dir, k, flag='Train') for i, k in zip(
        itertools.product(selected_CO_index_train, selected_Fore_index_train), range(130*70)))

Parallel(n_jobs=10)(delayed(split)(
    CO, beamed_fore, beamed_CO, i[0], i[1], base_dir, k, flag='Test') for i, k in zip(
        itertools.product(selected_CO_index_test, selected_Fore_index_test), range(32*26)))
# %%
