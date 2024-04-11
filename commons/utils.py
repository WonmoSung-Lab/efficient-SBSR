import numpy as np

def get_v(P1, P2):
    # get the LoR direction vector
    diff = P2 - P1
    norms = np.linalg.norm(diff, axis=0)
    v = diff / norms
    return v

def get_P1_v_times(LORs):
    # extract data from PET list mode data
    P1, P2, times = LORs.T[0:3, :], LORs.T[4:7, :], LORs.T[3, :]
    v = P2 - P1
    v = v / np.linalg.norm(v, axis=0)
    return P1, v, times

def get_knots(k , tc1, tc2, N):
    # calculate knots with padding
    T = np.linspace(tc1, tc2, np.maximum(2, N + 1 - k))   
    knots = np.r_[[0]*k, T, [T[-1]]*k]
    return knots

def get_euclidean(M, P1, P2):
    # get euclidian distances
    v = get_v(P1, P2)
    MP = P1 - M
    MP_v = (MP * v).sum(axis=0)
    return (MP ** 2).sum(axis=0) - (MP_v ** 2)