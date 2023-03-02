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
        i_img_color_mode = resNet(img[0], model)
        _, height, width, num_channels = i_img_color_mode.shape

    # if useHOG:
    #     img_hog, _ = hog_extraction(cv2.cvtColor(
    #         img[0], color_mode).astype(np.float64))
    #     img_hog = np.squeeze(img_hog)
    #     height, width, num_channels = img_hog.shape
    #     #height_hog, width_hog, num_channels_hog = img_hog.shape

    if color == True:
        num_channels = 11
    sigma = 2  # 10 king

    # ---------------- For HOG in all seperate RGB channels ------------------#
    # if useHOG:
    #     g_x = cv2.getGaussianKernel(height_hog, sigma)
    #     g_y = cv2.getGaussianKernel(width_hog, sigma)
    # else:
    # ---------
    g_x = cv2.getGaussianKernel(height, sigma)
    g_y = cv2.getGaussianKernel(width, sigma)

    g = np.outer(g_x, g_y)
    G = np.fft.fft2(g)

    # Preallocate number of indices needed

    # ---------------- For HOG in all seperate RGB channels ------------------#
    # if useHOG:
    #     all_A = [0]*num_channels_hog*3
    #     all_B = [0]*num_channels_hog*3
    #     all_F = [0]*num_channels_hog*3
    # else:
    # ------

    if useHOG == True:
        all_A = [0]*8
        all_B = [0]*8
        all_F = [0]*8
    else:
        all_A = [0]*num_channels
        all_B = [0]*num_channels
        all_F = [0]*num_channels

    for current_image in img:
        if useResNet == False:
            if useHOG:
                hog_image_pre = cv2.cvtColor(
                    current_image, color_mode).astype(np.float64)
                for i in range(num_channels):
                    hog_image_pre[:, :, i] = preprocessing(
                        hog_image_pre[:, :, i], width, height)
                hog_image_pre = cv2.resize(
                    hog_image_pre, (width-width % 8, height-height % 8))
                i_img_color_mode, _ = hog_extraction(hog_image_pre)

                i_img_color_mode = np.squeeze(i_img_color_mode)
                # i_img_color_mode = cv2.resize(
                #     i_img_color_mode, (width, height))
                h_hog, w_hog, hog_channels = i_img_color_mode.shape

            else:
                i_img_color_mode = cv2.cvtColor(
                    current_image, color_mode).astype(np.float64)
        else:
            i_img_color_mode = resNet(current_image, model)

        if color == True:
            i_img_color_mode = color_extraction(
                i_img_color_mode, mode="probability")

        for i in range(hog_channels if useHOG == True else num_channels):

            if useResNet == False:
                # ---------------- For HOG multichannel ------------------#
                if useHOG == True:
                    img_channel_norm = i_img_color_mode[:, :, i]
                    g_x = cv2.getGaussianKernel(h_hog, sigma)
                    g_y = cv2.getGaussianKernel(w_hog, sigma)

                    g = np.outer(g_x, g_y)
                    G = np.fft.fft2(g)
                else:
                    # ----
                    if len(i_img_color_mode.shape) == 2:
                        img_channel_norm = preprocessing(
                            i_img_color_mode, width, height)
                    else:
                        img_channel_norm = preprocessing(
                            i_img_color_mode[:, :, i], width, height)
            else:
                img_channel_norm = i_img_color_mode[0, :, :, i]

            # ---------------- For HOG in all seperate RGB channels ------------------#
            # if useHOG == True:
            #     img_channel_norm, hog_img = hog_extraction(img_channel_norm)
            #     img_channel_norm = np.squeeze(img_channel_norm)

            #     for j in range(num_channels_hog):
            #         F_i = np.fft.fft2(img_channel_norm[:,:,j])
            #         A = G * np.conjugate(F_i)
            #         B = F_i * np.conjugate(F_i)

            #         all_F[i*num_channels_hog+j] += F_i
            #         all_A[i*num_channels_hog+j] += A
            #         all_B[i*num_channels_hog+j] += B

            # else:
            # ---------

            F_i = np.fft.fft2(img_channel_norm)
            A = G * np.conjugate(F_i)
            B = F_i * np.conjugate(F_i)

            all_F[i] += F_i
            all_A[i] += A
            all_B[i] += B

    H_A_B = []
    H = 0
    # ALL A????!
    for a, b in zip(all_A, all_B):
        iH = a/(b+0.01)
        H += iH
        current = [iH, a, b]
        H_A_B.append(current)

    # for i in range(num_channels):
    #img_org = np.fft.fft2(cv2.cvtColor(img[0], color_mode)[:,:,i])
    # img_channel = img_org*H_A_B[i][0]
    #result_image += np.fft.ifft2(i_img_color_mode[0,:,:,i]).real
    # plt.imshow(result_image)
    # plt.show()

    # plt.imshow(np.fft.ifft2(H).real, cmap="gray")
    # plt.show()
    return H_A_B
