import cv2
import matplotlib.pyplot as plt
import numpy as np
from feature_extraction import hog_extraction
from funcitons import preprocessing

def filterInit(img, color_mode):
    height,width,num_channels = cv2.cvtColor(img[0], color_mode).astype(np.float64).shape
    sigma = 2 # 10 king

    # Create gaussian filter
    g_x = cv2.getGaussianKernel(height, sigma)
    g_y = cv2.getGaussianKernel(width, sigma)
    g = np.outer(g_x, g_y)
    G = np.fft.fft2(g)
  
    # Preallocate number of indices needed
    all_A = [0]*num_channels
    all_B = [0]*num_channels
    all_F = [0]*num_channels
    for current_image in img:
        i_img_color_mode = cv2.cvtColor(current_image, color_mode).astype(np.float64)
        for i in range(num_channels):
            img_channel_norm = preprocessing(i_img_color_mode[:,:,i], width, height)
            
            F_i = np.fft.fft2(img_channel_norm)

            A = G * np.conjugate(F_i)
            B = F_i * np.conjugate(F_i)
            
            all_F[i] += F_i
            all_A[i] += A
            all_B[i] += B

            # # HOG extraction - Use the feature vectors not the hog images
            # if channel == HOG:
            #     img_B = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV)[
            #         :, :, 0].astype(np.float64)
            #     img_G = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV)[
            #         :, :, 1].astype(np.float64)
            #     img_R = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV)[
            #         :, :, 2].astype(np.float64)

            #     normB = preprocessing(img_B, width, height)
            #     normG = preprocessing(img_G, width, height)
            #     normR = preprocessing(img_R, width, height)

                # fd, fd_B, fd_G, fd_R, imgB, imgG, imgR = hog_extraction(
                #     normB, normG, normR)

            # Color extraction
            # CODE::::
            # Concatenate the feature vectors of HOG and color extractions
 
    H_A_B = []
    H = 0
    for a,b in zip(A,B):
        iH = a/b
        H += iH
        current = [iH,a,b]
        H_A_B.append(current)
    result_image = 0
    for i in range(num_channels):
        img_org = np.fft.fft2(cv2.cvtColor(img[0], color_mode)[:,:,i])
        img_channel = img_org*H_A_B[i][0]
        result_image += np.fft.ifft2(img_channel).real
    plt.imshow(result_image)
    plt.show()
    # plt.imshow(np.fft.ifft2(H).real, cmap="gray")
    # plt.show()
    return H_A_B


