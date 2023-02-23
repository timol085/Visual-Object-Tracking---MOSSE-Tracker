import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from feature_extraction import hog_extraction

def filterInit(img,gray_scale=False):
    height, width, _ = img[0].shape
    center_y = height//2
    center_x = width // 2 
    sigma = 1 #10 king
    
    # Create gaussian filter
    g_x = cv2.getGaussianKernel(height, sigma)
    g_y = cv2.getGaussianKernel(width, sigma)
    g = np.outer(g_x, g_y)


    # # create a rectangular grid out of two given one-dimensional arrays
    # xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    # # calculating distance of each pixel from roi center
    # dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2*sigma)
        
    # create a rectangular grid out of two given one-dimensional arrays
    # xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    # # calculating distance of each pixel from roi center
    # dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2*sigma)
        
    # response = np.exp(-dist)
    # g = (response - response.min()) / (response.max() - response.min())

    G = np.fft.fft2(g)

    # Calculate MOSSE filter
    #f = np.fft.fft2(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY))

    A = 0#G * np.conj(f)
    B = 0#f * np.conj(f)
    A_B = 0
    A_G = 0
    A_R = 0
    B_B = 0
    B_G = 0
    B_R = 0
    for i in range(0, len(img)):
        if gray_scale:
            log_img = np.log(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY).astype(np.float64)+1)
            mean, std = np.mean(log_img), np.std(log_img)
            img_norm = (log_img - mean) / std

            window_col = np.hanning(width)
            window_row = np.hanning(height)
            col_mask, row_mask = np.meshgrid(window_col, window_row)
            window = col_mask * row_mask
            img_norm = img_norm * window
        
            F_i = np.fft.fft2(img_norm)

            #F_i = np.fft.fft2(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY))
            A += G * np.conjugate(F_i)
            B += F_i * np.conjugate(F_i)

        else:

            
            log_B = np.log(cv2.cvtColor(img[i],cv2.COLOR_BGR2HSV)[:,:,0].astype(np.float64)+1)
            log_G = np.log(cv2.cvtColor(img[i],cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float64)+1)
            log_R = np.log(cv2.cvtColor(img[i],cv2.COLOR_BGR2HSV)[:,:,2].astype(np.float64)+1)
    

            meanB, stdB = np.mean(log_B), np.std(log_B)
            meanG, stdG = np.mean(log_G), np.std(log_G)
            meanR, stdR = np.mean(log_R), np.std(log_R)
            normB = (log_B - meanB) / stdB
            normG = (log_G - meanG) / stdG
            normR = (log_R - meanR) / stdR
           
            # HOG extraction
            fd, fd_B, fd_G, fd_R, imgB, imgG, imgR = hog_extraction(normB,normG,normR)

            F_Bi = np.fft.fft2(imgB)
            F_Gi = np.fft.fft2(imgG)
            F_Ri = np.fft.fft2(imgR)

            A_B += G * np.conjugate(F_Bi)
            A_G += G * np.conjugate(F_Gi)
            A_R += G * np.conjugate(F_Ri)
            
            B_B += F_Bi * np.conjugate(F_Bi)
            B_G += F_Gi * np.conjugate(F_Gi)
            B_R += F_Ri * np.conjugate(F_Ri)
            

    if not gray_scale:
        A = A_B + A_G + A_R 
        B = B_B + B_G + B_R

            

    H = A/B

    # Test original image
    if gray_scale:

        img_org = np.fft.fft2(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY))

        # log_img = np.log(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)+1)
        # mean, std = np.mean(log_img), np.std(log_img)
        # img_norm = (log_img - mean) / std
        # img_org = np.fft.fft2(img_norm)

        result_img_org = img_org * H
        result_img_org = np.fft.ifft2(result_img_org).real

        plt.imshow(result_img_org,cmap="gray")
        plt.show()
    else:
        img_org_B = np.fft.fft2(cv2.cvtColor(img[0], cv2.COLOR_BGR2HSV)[:,:,0])
        img_org_R = np.fft.fft2(cv2.cvtColor(img[0], cv2.COLOR_BGR2HSV)[:,:,1])
        img_org_G = np.fft.fft2(cv2.cvtColor(img[0], cv2.COLOR_BGR2HSV)[:,:,2])

        # log_img = np.log(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)+1)
        # mean, std = np.mean(log_img), np.std(log_img)
        # img_norm = (log_img - mean) / std
        # img_org = np.fft.fft2(img_norm)

        result_img_org_B = img_org_B * H
        result_img_org_G = img_org_G * H
        result_img_org_R = img_org_R * H
        result_img_org_B = np.fft.ifft2(result_img_org_B).real
        result_img_org_G = np.fft.ifft2(result_img_org_G).real
        result_img_org_R = np.fft.ifft2(result_img_org_R).real
        result_img_org = result_img_org_B + result_img_org_G + result_img_org_R

        plt.imshow(result_img_org)
        plt.show()

    if gray_scale:
        plt.imshow(np.fft.ifft2(H).real,cmap="gray")
        plt.show()
        return H, A, B # H: Filter A/B, 
    else:
        return [[A_B/B_B,A_B,B_B], [A_G/B_G,A_G,B_G], [A_R/B_R,A_R,B_R]]
