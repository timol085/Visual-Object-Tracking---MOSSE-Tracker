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


def updateWindow(x_org, y_org, w_org, h_org, img, n_times_occluded,useResnet, useHOG, width_hog, height_hog,thr=7):
    peak, psr = get_peak_and_psr(np.fft.ifft2(img).real, useResnet,useHOG)
    
    if useResnet ==True:
        factorX= w_org/img.shape[1]
        factorY= h_org/img.shape[0]
        if psr > thr:
            dx = peak[1] - (img.shape[1] / 2)*factorX
            dy = peak[0] - (img.shape[0] / 2)*factorY
        else:
            dx = (peak[1] - (img.shape[1] / 2))*factorX
            dy = (peak[0] - (img.shape[0] / 2))*factorY
    else:

        if useHOG == True:
            factorX= w_org / width_hog
            factorY= h_org/height_hog
            peak= (peak[0]*factorY, peak[1]*factorX)
        # If PSR < 7 then the object may be occluded
        if psr > thr:
    
            dx = peak[1] - (w_org / 2)
            dy = peak[0] - (h_org / 2)
        else:
            dx = peak[1] - (w_org / 2)
            dy = peak[0] - (h_org / 2)
                    
            n_times_occluded[0] += 1        
        
    return int(x_org + dx), int(y_org + dy)
