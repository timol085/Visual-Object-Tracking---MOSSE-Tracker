"""Function to update the filter"""
import numpy as np


def updateFilter(Ai, Bi, Fi, Gi, eta=0.125):
    eta_Gi = eta*Gi
    eta_Fi = eta*Fi
    Ai = np.multiply(eta_Gi, np.conj(Fi)) + (1 - eta) * Ai
    Bi = np.multiply(eta_Fi, np.conj(Fi)) + (1 - eta) * Bi

    Hi = Ai / Bi

    return Hi, Ai, Bi
