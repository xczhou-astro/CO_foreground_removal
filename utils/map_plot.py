import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import eigh
import os


def PCA(arr, n=1):
    shape = arr.shape
    arr = arr.reshape(-1, arr.shape[-1])
    cov = arr.T @ arr
    w, v = eigh(cov)
    base = v[:, -n:]
    fg = arr@base
    fg = fg@base.T
    sig = arr - fg
    sig = sig.reshape(shape)
    return sig


idx = 2

test_dir = '../../Dataset/Data_corr_split/Test'

test_file_ls = os.listdir(test_dir)

test_filenames = np.array([os.path.join(test_dir, name)
                           for name in test_file_ls])[idx]

file = np.load(test_filenames)

show_maps = []
for i in range(3):
    show_maps.append(file['src'])
    show_maps.append(np.clip(PCA(file['src'], i + 1), None, 10))
    dir = 'result_' + str(i + 1)
    gen_maps = np.load(dir + '/gen_maps.npy')
    show_maps.append(np.clip(gen_maps[idx], None, 10))
    show_maps.append(np.clip(file['tar'], None, 10))

# show_maps: 3 * 4

titles = ['CO + Foregrounds [$\mu$K]', 'PCA-1 [$\mu$K]',
          'UNet [$\mu$K]', 'CO Signal [$\mu$K]']
titer = iter(titles)

pti = ['PCA-1 [$\mu$K]', 'PCA-2', 'PCA-3']
piter = iter(pti)

r = np.linspace(0, 7.449/2, 3)
l = np.linspace(-7.449/2, 0, 2, endpoint=False)
xaxis = np.around(np.array(l.tolist() + r.tolist()), 1)

yaxis = np.around(np.array(r[::-1].tolist() + l[::-1].tolist()), 1)

plt.figure(figsize=(20, 14))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    ax = plt.gca()
    img = show_maps[i][:, :, -1]
    vmin = np.min(img) - 1e-4
    im = ax.imshow(img, vmin=vmin)
    ax.set_xticks(np.linspace(0, 128, 5, endpoint=True), xaxis, fontsize=14)
    ax.set_yticks(np.linspace(0, 127, 5, endpoint=True), yaxis, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)

    if i >= 0 and i < 4:
        ax.set_title(next(titer), fontsize=20)

    if i % 4 == 1:
        ax.set_title(next(piter), fontsize=20)

    if (i + 1) % 4 == 1:
        ax.set_ylabel('Dec [$\degree$]', fontsize=20)
    if 9 <= (i + 1) <= 12:
        ax.set_xlabel('RA [$\degree$]', fontsize=20)

plt.tight_layout()
plt.savefig('Map_comparison_fiducial_00.pdf')
plt.show()
