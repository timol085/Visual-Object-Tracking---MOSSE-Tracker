import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def filterInit(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = imgGray.shape
    sigma = 2

    # Create gaussian filter
    g_x = cv2.getGaussianKernel(height, sigma)
    g_y = cv2.getGaussianKernel(width, sigma)
    g = np.outer(g_x, g_y)
    plt.imshow(g)
    plt.show()

    # Calculate filter
    G = np.fft.fft2(g)
    F = np.fft.fft2(imgGray)
    H = (G * np.conjugate(Img))/(F * np.conjugate(F))

    # Test
    Img = np.fft.fft2(cv2.cvtColor(
        cv2.imread('TSBB34_2.jpg'), cv2.COLOR_BGR2GRAY))
    G = H * Img
    G = np.fft.ifft2(G)
    plt.imshow(G.real)
    plt.show()


if __name__ == "__main__":
    img = cv2.imread('TSBB34_1.jpg')
    filterInit(img)
