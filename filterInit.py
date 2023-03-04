import cv2
import matplotlib.pyplot as plt
import numpy as np
from feature_extraction import hog_extraction, color_extraction
from funcitons import preprocessing
from resNet import resNet


def filterInit(img, color_mode, useResNet, useHOG, color, model):

    if useResNet == False:
        if len(cv2.cvtColor(img[0], color_mode).astype(np.float64).shape) == 2:
            num_channels = 1
            height, width = cv2.cvtColor(
                img[0], color_mode).astype(np.float64).shape
        else:
            height, width, num_channels = cv2.cvtColor(
                img[0], color_mode).astype(np.float64).shape
    else:
        i_img_color_mode = model(img[0])
        height, width, num_channels = i_img_color_mode.shape

    if useHOG:
        img_hog = cv2.cvtColor(img[0], color_mode).astype(np.float64)
        height_full,width_full,_ =img_hog.shape
        img_hog = cv2.resize(img_hog,(64,128))
        img_hog, _ = hog_extraction(img_hog)
        img_hog = np.squeeze(img_hog)
        height,width,num_channels = img_hog.shape
        
        

    if color == True:
        num_channels = 11
    sigma = 2  # 10 king

    g_x = cv2.getGaussianKernel(height, sigma)
    g_y = cv2.getGaussianKernel(width, sigma)

    g = np.outer(g_x, g_y)
    G = np.fft.fft2(g)

    # Preallocate number of indices needed
    all_A = [0]*num_channels
    all_B = [0]*num_channels
    all_F = [0]*num_channels

    for current_image in img:
        if useResNet==False:
            if useHOG:

                i_img_color_mode = cv2.cvtColor(current_image, color_mode).astype(np.float64)
                rgb_channels = 3
                for i in range(rgb_channels):
                    i_img_color_mode[:, :, i] = preprocessing(
                        i_img_color_mode[:, :, i], width_full, height_full)

                i_img_color_mode = cv2.resize(i_img_color_mode,(64,128))
                i_img_color_mode, _ = hog_extraction(i_img_color_mode)
                i_img_color_mode = np.squeeze(i_img_color_mode)
            else: 
                i_img_color_mode = cv2.cvtColor(current_image, color_mode).astype(np.float64)
        else:
            i_img_color_mode = model(current_image)

        if color == True:
            i_img_color_mode = color_extraction(
                i_img_color_mode, mode="probability")

        for i in range(num_channels):

            if useResNet == False:
                if useHOG == True:
                    img_channel_norm= i_img_color_mode[:,:,i]
                else: 
                    if len(i_img_color_mode.shape) == 2:
                        img_channel_norm = preprocessing(i_img_color_mode, width, height)
                    else:
                        img_channel_norm = preprocessing(i_img_color_mode[:,:,i], width, height)
            else:
                img_channel_norm = i_img_color_mode[:, :, i]
            
            F_i = np.fft.fft2(img_channel_norm)
            A = G * np.conjugate(F_i)
            B = F_i * np.conjugate(F_i)

            all_F[i] += F_i
            all_A[i] += A
            all_B[i] += B


    H_A_B = []
    H = 0
    for a,b in zip(all_A,all_B):
        iH = a/(b+0.01)
        H += iH
        current = [iH, a, b]
        H_A_B.append(current)
    result_image = 0

    # for i in range(num_channels):
    #img_org = np.fft.fft2(cv2.cvtColor(img[0], color_mode)[:,:,i])
    # img_channel = img_org*H_A_B[i][0]
    #result_image += np.fft.ifft2(i_img_color_mode[0,:,:,i]).real
    # plt.imshow(result_image)
    # plt.show()

    # plt.imshow(np.fft.ifft2(H).real, cmap="gray")
    # plt.show()
    return H_A_B
