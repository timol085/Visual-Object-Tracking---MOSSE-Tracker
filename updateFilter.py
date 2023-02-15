"""Function to update the filter"""
import numpy as np


def updateFilter(Ai, Bi, Fi, Gi, eta=0.125):
    eta_Gi = eta*Gi
    eta_Fi = eta*Fi
    Ai = np.multiply(eta_Gi, np.conj(Fi)) + (1 - eta) * Ai
    Bi = np.multiply(eta_Fi, np.conj(Fi)) + (1 - eta) * Bi

    Hi = Ai / Bi

    return Hi, Ai, Bi


def updateWindow(x_org, y_org, w_org, h_org, F, peak[]):
    dx = (w_org / 2) - peak[0]
    dy = (h_org / 2) - peak[1]

    return 0
