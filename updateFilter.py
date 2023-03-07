"""Function to update the filter"""
import numpy as np

from get_peak_and_psr import get_peak_and_psr
from funcitons import crop_image


def updateFilter(Ai, Bi, Fi, Gi, useResnet, eta=0.125):
    eta_Gi = eta*Gi
    eta_Fi = eta*Fi
    Ai = np.multiply(eta_Gi, np.conj(Fi)) + (1 - eta) * Ai
    Bi = 0.01+np.multiply(eta_Fi, np.conj(Fi)) + (1 - eta) * Bi

    Hi = Ai / Bi

    return Hi, Ai, Bi


def updateWindow(x_org, y_org, w_org, h_org, img, n_times_occluded,useResnet, useHOG, useColor,thr=5):
    peak, psr = get_peak_and_psr(np.fft.ifft2(img).real, useResnet,useHOG)
   
    isOkPsr = True
    
    if useResnet ==True or useHOG==True:
        thr= 2.5
        factorX= w_org/img.shape[1]
        factorY= h_org/img.shape[0]
        if psr > thr:
            dx = (peak[1] - (img.shape[1] / 2))*factorX
            dy = (peak[0] - (img.shape[0] / 2))*factorY
        else:
            isOkPsr=False
            dx,dy=(0,0)
    else:
        #If PSR < 7 then the object may be occluded
        if useColor == True:
            thr=3.5
            
        if psr > thr:    
            dx = peak[1] - (w_org / 2)
            dy = peak[0] - (h_org / 2)
        else:
            isOkPsr=False
            dx,dy=(0,0)
            n_times_occluded[0] += 1        
        
        #2.2 HOG
        #3.5 COLOR
        #3 Resnet
        #9 RGB 
        
        
    return isOkPsr,int(x_org + dx), int(y_org + dy)
