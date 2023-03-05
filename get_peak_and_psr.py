from matplotlib import pyplot as plt
import numpy as np
import cv2


def get_peak_and_psr(img, useResnet,useHOG):
    # Calculate peak value
    img = (img - np.min(img))/(np.max(img)-np.min(img))
    gmax = np.max(img)

    # Get indices of peak value
    peak_index = np.where(img == gmax)
    peak_x = peak_index[0][0]
    peak_y = peak_index[1][0]

    # Extract a 11x11 window around the peak
    if useResnet or useHOG:
        window_size = 1
    else: window_size=11
    
    x_start = max(peak_x - window_size // 2, 0)
    x_end = min(peak_x + window_size // 2 + 1, img.shape[0])
    y_start = max(peak_y - window_size // 2, 0)
    y_end = min(peak_y + window_size // 2 + 1, img.shape[1])

    # Calculate sidelobe by excluding the 11x11 window around the peak
    sidelobe = img
    sidelobe[x_start:x_end, y_start:y_end] = 0

    #plt.imshow(sidelobe)
    #plt.show()
    
    # Calculate mean and standard deviation of the sidelobe
    mean = np.mean(sidelobe)
    std = np.std(sidelobe)

    # Calculate PSR
    psr = (gmax - mean) / std

    return [peak_x, peak_y], psr
