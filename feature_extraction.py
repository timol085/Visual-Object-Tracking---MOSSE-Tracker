import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure
from matplotlib import pyplot as plt
import scipy.io
import os


# Feature extraction for color and edge features


def hog_extraction(norm_channel):
    # HOG feature extraction
    fd, hog_image = hog(norm_channel, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(
        1, 1), visualize=True, channel_axis=-1, feature_vector=False)

    # Reshape to shape of cropped img
    # cropped_shape = normB.shape
    # fd_B = fd_B.reshape(cropped_shape[:2])
    # fd_G = fd_G.reshape(cropped_shape[:2])
    # fd_R = fd_R.reshape(cropped_shape[:2])

    # #Show HOG img
    # print(fd)
    # fig, (ax1, ax2) = plt.subplots(
    #     1, 2, figsize=(8, 4), sharex=True, sharey=True)
    # ax1.axis('off')
    # ax1.imshow(norm_channel, cmap=plt.cm.gray)
    # ax1.set_title('Input image')

    # hog_image_rescaled = exposure.rescale_intensity(
    #     hog_image, in_range=(0, 10))
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

    return fd, hog_image


# Got from Gitlab Liu TSBB34 repo
COLOR_NAMES = ['black', 'blue', 'brown', 'grey', 'green', 'orange',
               'pink', 'purple', 'red', 'white', 'yellow']
COLOR_RGB = [[0, 0, 0], [0, 0, 1], [.5, .4, .25], [.5, .5, .5], [0, 1, 0], [1, .8, 0],
             [1, .5, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]]

COLORNAMES_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), 'colornames_w2c.mat')
COLORNAMES_TABLE = scipy.io.loadmat(COLORNAMES_TABLE_PATH)['w2c']


def color_extraction(image, mode="index"):
    image = image.astype('double')
    idx = np.floor(image[..., 0] / 8) + 32 * np.floor(image[...,
                                                            1] / 8) + 32 * 32 * np.floor(image[..., 2] / 8)
    m = COLORNAMES_TABLE[idx.astype('int')]

    if mode == 'index':
        return np.argmax(m, 2)
    elif mode == 'probability':
        return m
    else:
        raise ValueError("No such mode: '{}'".format(mode))
