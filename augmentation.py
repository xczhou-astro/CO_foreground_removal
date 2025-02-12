import numpy as np
import argparse
from joblib import Parallel, delayed
import os


parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str,
                    help='arr filename that to augment')

args = parser.parse_args()

filename = args.filename

print(filename)

file = np.load(filename)

print(file.shape)


def transform(arr):
    container = np.zeros(shape=(8, *arr.shape))
    container[0] = arr
    for i, k in enumerate(range(1, 4), start=1):
        container[i] = np.rot90(arr, k=k, axes=(0, 1))

    for i, a in enumerate(range(2), start=4):
        container[i] = np.flip(arr, axis=a)

    for i, a in enumerate(range(2), start=6):
        container[i] = np.rot90(np.flip(arr, axis=a), axes=(0, 1))

    return container


augmented_arr = Parallel(n_jobs=30)(delayed(transform)(file[i])
                                    for i in range(file.shape[0]))

augmented_arr = np.array(augmented_arr)
augmented_arr = np.reshape(augmented_arr, (-1, *file.shape[1:]))


print(augmented_arr.shape)

saved_filename = filename[:-4] + '_aug.npy'

assert not os.path.exists(saved_filename)

np.save(saved_filename, augmented_arr)
