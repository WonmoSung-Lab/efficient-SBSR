import numpy as np
from scipy.interpolate import BSpline
from .utils import get_v

def objf3D(k, s, knots, P1, P2, t, dmax, lambda0):
    n = len(t)
    s = s.reshape((3, -1))
    m = s.shape[1]

    v = get_v(P1=P1, P2=P2)
    
    s_x, s_y, s_z = s

    spl_x = BSpline(knots, s_x, k) 
    spl_y = BSpline(knots, s_y, k)
    spl_z = BSpline(knots, s_z, k) 

    M_x = spl_x(t).reshape((-1, len(t)))
    M_y = spl_y(t).reshape((-1, len(t)))
    M_z = spl_z(t).reshape((-1, len(t)))

    M = np.concatenate([M_x, M_y, M_z], axis=0)

    MP = M - P1
    dMPv = np.multiply(MP, v).sum(axis=0)
    dMPMP = np.multiply(MP, MP).sum(axis=0)
    df = dMPMP - np.multiply(dMPv, dMPv)

    dmax_2 = np.repeat(dmax**2, len(df))
    D = np.mean(np.minimum(dmax**2, df))

    s_diff = s[:, :-1] - s[:, 1:]
    sum_s_diff2 = np.multiply(s_diff, s_diff).sum(axis=0)

    R = np.mean(sum_s_diff2)

    J = D + lambda0 * R

    return J