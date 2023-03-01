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

    if color == True:
        num_channels = 11
    sigma = 2  # 10 king

    # Create gaussian filter
    g_x = cv2.getGaussianKernel(height, sigma)
    g_y = cv2.getGaussianKernel(width, sigma)
    g = np.outer(g_x, g_y)
    G = np.fft.fft2(g)

    # Preallocate number of indices needed
    all_A = [0]*num_channels
    all_B = [0]*num_channels
    all_F = [0]*num_channels

    # HOG
    fd = 0

    for current_image in img:
        if useResNet == False:
            i_img_color_mode = cv2.cvtColor(
                current_image, color_mode).astype(np.float64)
        else:
            i_img_color_mode = resNet(current_image, model)

        if color == True:
            i_img_color_mode = color_extraction(
                i_img_color_mode, mode="probability")

        for i in range(num_channels):

            if useResNet == False:
                if len(i_img_color_mode.shape) == 2:
                    img_channel_norm = preprocessing(
                        i_img_color_mode, width, height)
                else:
                    img_channel_norm = preprocessing(
                        i_img_color_mode[:, :, i], width, height)
            else:
                img_channel_norm = i_img_color_mode[0, :, :, i]

              # HOG extraction - Use the feature vectors not the hog images
            fd = None
            if useHOG == True:
                for i in range(num_channels):
                    hog_image_pre = cv2.cvtColor(
                        current_image, color_mode).astype(np.float64)[:, :, i]
                    hog_image_pre = cv2.resize(
                        hog_image_pre, (width*4, height*4))
                    img_channel_norm, hog_img = hog_extraction(hog_image_pre)
                    img_channel_norm = img_channel_norm.squeeze()
                    if fd is None:
                        fd = img_channel_norm
                    else:
                        fd = np.concatenate([img_channel_norm, fd])
                        img_channel_norm = fd
                    # img_channel_norm = np.squeeze(img_channel_norm)

                    print(f"G_{i}: ", G.shape)
                    print(f"fd_{i}: ", img_channel_norm.shape)
                _, _, hog_channels = img_channel_norm.shape
                all_A = [0]*hog_channels
                all_B = [0]*hog_channels
                all_F = [0]*hog_channels

                for i in range(hog_channels):
                    F_i = np.fft.fft2(img_channel_norm[:, :, i])
                    A = G * np.conjugate(F_i)
                    B = F_i * np.conjugate(F_i)

                    all_F[i] += F_i
                    all_A[i] += A
                    all_B[i] += B

            if useHOG == False:
                F_i = np.fft.fft2(img_channel_norm)
                A = G * np.conjugate(F_i)
                B = F_i * np.conjugate(F_i)

                all_F[i] += F_i
                all_A[i] += A
                all_B[i] += B

            # Color extraction
            # CODE::::
            # Concatenate the feature vectors of HOG and color extractions

    H_A_B = []
    H = 0
    # ALL A????!
    for a, b in zip(all_A, all_B):
        iH = a/(b+0.01)
        H += iH
        current = [iH, a, b]
        H_A_B.append(current)
    result_image = 0
    print("HAB: ", H_A_B)

    # for i in range(num_channels):
    #img_org = np.fft.fft2(cv2.cvtColor(img[0], color_mode)[:,:,i])
    # img_channel = img_org*H_A_B[i][0]
    #result_image += np.fft.ifft2(i_img_color_mode[0,:,:,i]).real
    # plt.imshow(result_image)
    # plt.show()

    # plt.imshow(np.fft.ifft2(H).real, cmap="gray")
    # plt.show()
    return H_A_B
