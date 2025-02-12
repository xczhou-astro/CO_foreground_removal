# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import itertools

# %%
CO = np.load('data_low/CO_signal_aug.npy')
beamed_fore = np.load('data/beamed_foregrounds_aug.npy')
beamed_CO = np.load('data_low/beamed_CO_signal_aug.npy')


# %%
def split(signal, b_fg, b_sig, signal_idx, fg_idx, k=0, flag='Train'):
    co_signal = signal[signal_idx]
    b_signal = b_sig[signal_idx]

    mixed = b_signal + b_fg[fg_idx]

    saved_filename = 'Dataset/Data_low_split_aug/' + \
        flag + '/arr_{:05d}.npz'.format(k)

    np.savez_compressed(saved_filename, src=mixed, tar=co_signal)


os.makedirs('Dataset/Data_low_split_aug/Train', exist_ok=True)
os.makedirs('Dataset/Data_low_split_aug/Test', exist_ok=True)

np.random.seed(1234)
selected_CO_index_train = np.random.choice(
    np.arange(CO.shape[0]), 300, replace=False)
np.random.seed(4321)
selected_Fore_index_train = np.random.choice(
    np.arange(beamed_fore.shape[0]), 300, replace=False)

CO_index_res = np.setdiff1d(np.arange(CO.shape[0]), selected_CO_index_train)
np.random.seed(1234)
selected_CO_index_test = np.random.choice(CO_index_res, 100, replace=False)

Fore_index_res = np.setdiff1d(
    np.arange(beamed_fore.shape[0]), selected_Fore_index_train)
np.random.seed(4321)
selected_Fore_index_test = np.random.choice(Fore_index_res, 100, replace=False)

Parallel(n_jobs=30)(delayed(split)(
    CO, beamed_fore, beamed_CO, i[0], i[1], k, flag='Train') for i, k in zip(
        itertools.product(selected_CO_index_train, selected_Fore_index_train), range(300*300)))

Parallel(n_jobs=30)(delayed(split)(
    CO, beamed_fore, beamed_CO, i[0], i[1], k, flag='Test') for i, k in zip(
        itertools.product(selected_CO_index_test, selected_Fore_index_test), range(100*100)))
# %%
