# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from power_spectrum import power_spectrum
from scipy.linalg import eigh
import argparse

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


parser = argparse.ArgumentParser()
parser.add_argument('--pca_n', type=int,
                    help='pca_number')

args = parser.parse_args()

pca_n = args.pca_n


def encoding(layer_in, f_1x1_r, f_out):

    e1 = layers.BatchNormalization()(layer_in)
    e2 = layers.LeakyReLU()(e1)
    e3 = layers.Conv2D(f_1x1_r, 1, strides=2,
                       padding='same', kernel_initializer='he_normal')(e2)
#    e4=layers.Dropout(drop_rate)(e3, training=True)
    e5 = layers.BatchNormalization()(e3)
    e6 = layers.LeakyReLU()(e5)
    e7 = layers.Conv2D(f_out, 3, strides=1,
                       padding='same', kernel_initializer='he_normal')(e6)
#    e8=layers.Dropout(drop_rate)(e7, training=True)

    skip = layers.Conv2D(f_out, 1, strides=2,
                         padding='same', kernel_initializer='he_normal')(layer_in)

    add = layers.Add()([e7, skip])

    return add


def encoding_nopool(layer_in, f_1x1):

    f_out = layer_in.shape[-1]

    e1 = layers.BatchNormalization()(layer_in)
    e2 = layers.LeakyReLU()(e1)
    e3 = layers.Conv2D(f_1x1, 1, strides=1,
                       padding='same', kernel_initializer='he_normal')(e2)
#    e4=layers.Dropout(drop_rate)(e3, training=True)
    e5 = layers.BatchNormalization()(e3)
    e6 = layers.LeakyReLU()(e5)
    e7 = layers.Conv2D(f_out, 3, strides=1,
                       padding='same', kernel_initializer='he_normal')(e6)
#    e8=layers.Dropout(drop_rate)(e7, training=True)

    add = layers.Add()([e7, layer_in])

    return add


def decoding(layer_in, skip_block, f_1x1_r, f_out):

    d1 = layers.BatchNormalization()(layer_in)
    d2 = layers.LeakyReLU()(d1)
    d3 = layers.Conv2D(
        f_1x1_r, 1, strides=1, padding='same', kernel_initializer='he_normal')(d2)
#    d4=layers.Dropout(drop_rate)(d3, training=True)
    d5 = layers.BatchNormalization()(d3)
    d6 = layers.LeakyReLU()(d5)

    up = layers.UpSampling2D()(d6)
    d7 = layers.Conv2D(
        f_out, 3, strides=1, padding='same', kernel_initializer='he_normal')(up)
#    d8=layers.Dropout(drop_rate)(d7, training=True)

    up_skip = layers.UpSampling2D()(layer_in)
    skip = layers.Conv2D(
        f_out, 1, strides=1, padding='same', kernel_initializer='he_normal')(up_skip)

    add = layers.Add()([d7, skip])

    outputs = layers.concatenate([add, skip_block], axis=-1)

    return outputs


def decoding_nopool(layer_in, f_1x1):

    f_out = layer_in.shape[-1]

    d1 = layers.BatchNormalization()(layer_in)
    d2 = layers.LeakyReLU()(d1)
    d3 = layers.Conv2D(
        f_1x1, 1, strides=1, padding='same', kernel_initializer='he_normal')(d2)
#    d4=layers.Dropout(drop_rate)(d3, training=True)
    d5 = layers.BatchNormalization()(d3)
    d6 = layers.LeakyReLU()(d5)
    d7 = layers.Conv2D(
        f_out, 3, strides=1, padding='same', kernel_initializer='he_normal')(d6)
#    d8=layers.Dropout(drop_rate)(d7, training=True)

    add = layers.Add()([d7, layer_in])

    return add


def build_generator():
    inputs = layers.Input(shape=(128, 128, 51))

    conv = layers.Conv2D(64, 3, strides=1, padding='same')(inputs)
    conv = layers.LeakyReLU()(conv)  # 128 128 64 4

    enc_reduce_1 = encoding(conv, 32, 128)  # 64 64 32
    enc_nopool_1 = encoding_nopool(enc_reduce_1, 64)  # 8

    enc_reduce_2 = encoding(enc_nopool_1, 64, 256)  # 32 32 16
    enc_nopool_2 = encoding_nopool(enc_reduce_2, 128)  # 16

    enc_reduce_3 = encoding(enc_nopool_2, 128, 512)  # 16 16 8
    enc_nopool_3 = encoding_nopool(enc_reduce_3, 256)  # 32

    enc_reduce_4 = encoding(enc_nopool_3, 256, 1024)  # 8 8 4
    enc_nopool_4 = encoding_nopool(enc_reduce_4, 512)  # 64

    enc_reduce_5 = encoding(enc_nopool_4, 256, 1024)
    enc_nopool_5 = encoding_nopool(enc_reduce_5, 512)

    dec_up_1 = decoding(enc_nopool_5, enc_nopool_4, 512, 512)  # 16 16 8 32+32
    dec_noup_1 = decoding_nopool(dec_up_1, 512)  # 64

    dec_up_2 = decoding(dec_noup_1, enc_nopool_3, 512, 512)  # 32 32 16 16+16
    dec_noup_2 = decoding_nopool(dec_up_2, 512)  # 32

    dec_up_3 = decoding(dec_noup_2, enc_nopool_2, 256, 256)  # 64 64 32 8+8
    dec_noup_3 = decoding_nopool(dec_up_3, 256)  # 16

    dec_up_4 = decoding(dec_noup_3, enc_nopool_1, 128, 128)  # 128 128 64 4+4
    dec_noup_4 = decoding_nopool(dec_up_4, 128)  # 8

    dec_up_5 = decoding(dec_noup_4, conv, 64, 64)
    dec_noup_5 = decoding_nopool(dec_up_5, 64)     

    out = layers.Conv2D(
        51, 3, strides=1, padding='same')(dec_noup_5)  # 128 128 64 1

    outputs = out

    model = tf.keras.models.Model(inputs, outputs)
    return model


#build_generator().summary()
# %%


# def noise(arr):
#     return arr + np.random.normal(scale=sigma, size=arr.shape)


def PCA(arr, pca_n=1):
    shape = arr.shape
    arr = arr.reshape(-1, arr.shape[-1])
    cov = arr.T @ arr
    w, v = eigh(cov)
    base = v[:, -pca_n:]
    fg = arr@base
    fg = fg@base.T
    sig = arr - fg
    sig = sig.reshape(shape)
    return sig


# def pca_noise(arr):
#     return PCA(noise(arr)).astype('float32')


class Monitor(tf.keras.callbacks.Callback):

    def __init__(self, filenames, num_img=3):
        self.num_img = num_img
        self.filenames = filenames

    def on_epoch_end(self, epoch, logs=None):
        choice = np.random.choice(np.arange(self.filenames.shape[0]),
                                  self.num_img, replace=False)
        src = []
        tar = []
        gen = []
        for index in choice:
            file = np.load(self.filenames[index])
            source = PCA(file['src'], pca_n)
            target = file['tar']
            src.append(source)
            tar.append(target)

        src = np.array(src)
        tar = np.array(tar)
        gen = self.model(src)

        plt.figure(figsize=(12, 16))
        for i in range(self.num_img):
            plt.subplot(4, 3, i + 1)
            plt.imshow(src[i, :, :, 0])
            plt.axis('off')
            plt.colorbar()

        for i in range(self.num_img):
            plt.subplot(4, 3, i + 4)
            plt.imshow(tar[i, :, :, 0])
            plt.axis('off')
            plt.colorbar()

        for i in range(self.num_img):
            plt.subplot(4, 3, i + 7)
            plt.imshow(gen[i, :, :, 0])
            plt.axis('off')
            plt.colorbar()

        for i in range(self.num_img):
            k_src, p_src, s_src = power_spectrum(
                src[i], 300, 300)
            k_tar, p_tar, s_tar = power_spectrum(
                tar[i], 300, 300)
            k_gen, p_gen, s_gen = power_spectrum(
                gen[i], 300, 300)


            plt.subplot(4, 3, i + 10)
            plt.errorbar(k_src, p_src, yerr=np.sqrt(s_src **
                             2), fmt='.', label='Src')
            plt.errorbar(k_tar, p_tar, yerr=np.sqrt(s_tar **
                             2), fmt='.', label='Tar')
            plt.errorbar(k_gen, p_gen, yerr=np.sqrt(s_gen **
                             2), fmt='.', label='Gen')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()

        plt.savefig('logs/imgs_{:0>3d}'.format(epoch))
        plt.close()
        self.model.save('logs/model_{:0>3d}'.format(epoch))


model = build_generator()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='logcosh')


def read_npz_file(filename):
    file = np.load(filename)
    return PCA(file['src'], pca_n).astype('float32'), file['tar'].astype('float32')



@tf.function
def map_function(filename):
    return tf.numpy_function(read_npz_file, [filename], [tf.float32, tf.float32])



os.makedirs('logs', exist_ok=True)

train_dir = '../../Dataset/Data_corr_split/Train'
test_dir = '../../Dataset/Data_corr_split/Test'

train_file_ls = os.listdir(train_dir)
test_file_ls = os.listdir(test_dir)

train_filenames = np.array([os.path.join(train_dir, name)
                           for name in train_file_ls])
test_filenames = np.array([os.path.join(test_dir, name)
                          for name in test_file_ls])



train_ds = tf.data.Dataset.from_tensor_slices(train_filenames).map(
        map_function, num_parallel_calls=tf.data.AUTOTUNE).batch(8)
test_ds = tf.data.Dataset.from_tensor_slices(test_filenames).map(
        map_function, num_parallel_calls=tf.data.AUTOTUNE).batch(8)

cbk = Monitor(test_filenames, num_img=3)

rlp = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

his = model.fit(train_ds, epochs=30, verbose=2,
                validation_data=test_ds, callbacks=[cbk, rlp])

plt.figure(figsize=(8, 6))
plt.plot(his.history['loss'], label='Training')
plt.plot(his.history['val_loss'], label='Validation')
plt.legend()
plt.savefig('logs/Losses')
