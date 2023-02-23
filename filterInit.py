import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from feature_extraction import hog_extraction
from funcitons import preprocessing

global GRAY_SCALE
global RGB
global HOG
global RESNET
GRAY_SCALE = 0
RGB = 1
HOG = 2
RESNET = 3

def filterInit(img, channel):
    height, width, num_channels = img[0].shape
    center_y = height//2
    center_x = width // 2 
    sigma = 2 #10 king
    
    # Create gaussian filter
    g_x = cv2.getGaussianKernel(height, sigma)
    g_y = cv2.getGaussianKernel(width, sigma)
    g = np.outer(g_x, g_y)
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
        if channel == GRAY_SCALE:
           # Call preprocessing
           img_gray = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
           img_norm = preprocessing(img_gray, width, height)
           
           F_i = np.fft.fft2(img_norm)
           #F_i = np.fft.fft2(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY))
           A += G * np.conjugate(F_i)
           B += F_i * np.conjugate(F_i)

        else:
            # Preprocessing - if/Switch-case to handle different multichannel cases
            if channel == RGB:
                img_B = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.float64)
                img_G = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float64)
                img_R = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float64)
                
                normB = preprocessing(img_B, width, height)
                normG = preprocessing(img_G, width, height)
                normR = preprocessing(img_R, width, height)
        
                F_Bi = np.fft.fft2(normB)
                F_Gi = np.fft.fft2(normG)
                F_Ri = np.fft.fft2(normR)
    
            # HOG extraction - Use the feature vectors not the hog images 
            if channel == HOG:
                img_B = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.float64)
                img_G = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float64)
                img_R = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float64)
                
                normB = preprocessing(img_B, width, height)
                normG = preprocessing(img_G, width, height)
                normR = preprocessing(img_R, width, height)
                
                fd, fd_B, fd_G, fd_R, imgB, imgG, imgR = hog_extraction(normB,normG,normR)
                F_Bi = np.fft.fft2(imgB)
                F_Gi = np.fft.fft2(imgG)
                F_Ri = np.fft.fft2(imgR)
            
            # Color extraction
            # CODE::::
            #Concatenate the feature vectors of HOG and color extractions 

            A_B += G * np.conjugate(F_Bi)
            A_G += G * np.conjugate(F_Gi)
            A_R += G * np.conjugate(F_Ri)
            
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

    if channel ==   GRAY_SCALE:
        plt.imshow(np.fft.ifft2(H).real,cmap="gray")
        plt.show()
        return H, A, B # H: Filter A/B, 
    else:
        return [[A_B/B_B,A_B,B_B], [A_G/B_G,A_G,B_G], [A_R/B_R,A_R,B_R]]
