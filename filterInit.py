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

    # Test target moved
    img_test = np.fft.fft2(cv2.cvtColor(cv2.imread('TSBB34_2.jpg'), cv2.COLOR_BGR2GRAY))
    result_img_test = img_test * H
    result_img_test = np.fft.ifft2(result_img_test).real
    plt.imshow(result_img_test)
    plt.show()


if __name__ == "__main__":
    images = [cv2.imread('TSBB34_1.jpg'),cv2.imread('TSBB34_1_mod2.jpg'),cv2.imread('TSBB34_1_mod3.jpg'),cv2.imread('TSBB34_1_mod4.jpg'),cv2.imread('TSBB34_1_mod5.jpg'),cv2.imread('TSBB34_1_mod6.jpg'),cv2.imread('TSBB34_1_mod7.jpg')]
    #img = cv2.imread('TSBB34_1.jpg')
    filterInit(images)
