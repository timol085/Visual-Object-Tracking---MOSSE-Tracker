import cv2
import matplotlib.pyplot as plt
import numpy as np
from feature_extraction import hog_extraction
from funcitons import preprocessing
from resNet import resNet

def filterInit(img, color_mode, useResNet, model):
    
    
    if useResNet==False:
        height,width,num_channels = cv2.cvtColor(img[0], color_mode).astype(np.float64).shape
    else:
        i_img_color_mode = resNet(img[0], model)
        _,height, width,num_channels= i_img_color_mode.shape
        
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
        if useResNet==False:
            i_img_color_mode = cv2.cvtColor(current_image, color_mode).astype(np.float64)
        else:
            i_img_color_mode = resNet(current_image, model)
        
        
        for i in range(num_channels):
            
            if useResNet==False:
                img_channel_norm = preprocessing(i_img_color_mode[:,:,i], width, height)
            else:
                img_channel_norm= i_img_color_mode[0,:,:,i]
            
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
    #ALL A????!
    for a,b in zip(all_A,all_B):
        iH = a/(b+0.01)
        H += iH
        current = [iH,a,b]
        H_A_B.append(current)
    result_image = 0
    
    #for i in range(num_channels):
        #img_org = np.fft.fft2(cv2.cvtColor(img[0], color_mode)[:,:,i])
       # img_channel = img_org*H_A_B[i][0]
        #result_image += np.fft.ifft2(i_img_color_mode[0,:,:,i]).real
    #plt.imshow(result_image)
    #plt.show()
    
    # plt.imshow(np.fft.ifft2(H).real, cmap="gray")
    # plt.show()
    return H_A_B


    B_B += F_Bi * np.conjugate(F_Bi)
    B_G += F_Gi * np.conjugate(F_Gi)
    B_R += F_Ri * np.conjugate(F_Ri)

    if channel != GRAY_SCALE:
        A = A_B + A_G + A_R
        B = B_B + B_G + B_R

    H = A/B

    # Test original image
    if channel == GRAY_SCALE:

        img_org = np.fft.fft2(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY))

        # log_img = np.log(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)+1)
        # mean, std = np.mean(log_img), np.std(log_img)
        # img_norm = (log_img - mean) / std
        # img_org = np.fft.fft2(img_norm)

        result_img_org = img_org * H
        result_img_org = np.fft.ifft2(result_img_org).real

        plt.imshow(result_img_org, cmap="gray")
        plt.show()
    else:
        img_org_B = np.fft.fft2(cv2.cvtColor(
            img[0], cv2.COLOR_BGR2HSV)[:, :, 0])
        img_org_R = np.fft.fft2(cv2.cvtColor(
            img[0], cv2.COLOR_BGR2HSV)[:, :, 1])
        img_org_G = np.fft.fft2(cv2.cvtColor(
            img[0], cv2.COLOR_BGR2HSV)[:, :, 2])

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

    if channel == GRAY_SCALE:

        return H, A, B  # H: Filter A/B,
    else:
        return [[A_B/B_B, A_B, B_B], [A_G/B_G, A_G, B_G], [A_R/B_R, A_R, B_R]]
