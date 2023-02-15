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
    A = 0
    B = 0
    for i in range(0, len(img)):
        F_i = np.fft.fft2(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY))
        A = A + G * np.conjugate(F_i)
        B = B + F_i * np.conjugate(F_i)

    H = A/B

    # Test original image
    img_org = np.fft.fft2(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY))
    result_img_org = img_org * H
    result_img_org = np.fft.ifft2(result_img_org).real
    plt.imshow(result_img_org)
    plt.show()

    return H, A, B
