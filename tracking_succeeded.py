import numpy as np
import cv2

def tracking_succeeded(img,min_psr):
    # Calculate peak value
    gmax = np.max(img)

    # Get indices of peak value
    peak_index = np.where(img == gmax)
    peak_x = peak_index[0][0]
    peak_y = peak_index[1][0]

    # Extract a 11x11 window around the peak
    window_size = 11
    x_start = max(peak_x - window_size // 2, 0)
    x_end = min(peak_x + window_size // 2 + 1, img.shape[0])
    y_start = max(peak_y - window_size // 2, 0)
    y_end = min(peak_y + window_size // 2 + 1, img.shape[1])

    # Calculate the sidelobe by excluding the 11x11 window around the peak
    sidelobe = img
    sidelobe[x_start:x_end, y_start:y_end] = 0

    # Calculate the mean and standard deviation of the sidelobe
    mean = np.mean(sidelobe)
    std = np.std(sidelobe)

    # Calculate the PSR
    psr = (gmax - mean) / std

    if psr > min_psr:
        return  True
    return False

if __name__ == "__main__":
    img = cv2.imread('GaussianTest.jpg')
    print(tracking_succeeded(img, 8))
    
