import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def filterInit(img):
    height, width, _ = img[0].shape
    sigma = 2

    # Create gaussian filter
    g_x = cv2.getGaussianKernel(height, sigma)
    g_y = cv2.getGaussianKernel(width, sigma)
    g = np.outer(g_x, g_y)

    G = np.fft.fft2(g)

    # Calculate MOSSE filter
    f = np.fft.fft2(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY))

    A = 0#G * np.conj(f)
    B = 0#f * np.conj(f)
    for i in range(0, len(img)):
        log_img = np.log(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY).astype(np.float64)+1)
        mean, std = np.mean(log_img), np.std(log_img)
        img_norm = (log_img - mean) / std
        F_i = np.fft.fft2(img_norm)

        #F_i = np.fft.fft2(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY))
        A += G * np.conjugate(F_i)
        B += F_i * np.conjugate(F_i)

    H = A/B

    # Test original image
    img_org = np.fft.fft2(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY))

    # log_img = np.log(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)+1)
    # mean, std = np.mean(log_img), np.std(log_img)
    # img_norm = (log_img - mean) / std
    # img_org = np.fft.fft2(img_norm)

    result_img_org = img_org * H
    result_img_org = np.fft.ifft2(result_img_org).real

    plt.imshow(result_img_org)
    plt.show()

    return H, A, B
