import numpy as np
from scipy.optimize import minimize

from .backprop import CalculateLoss
from .utils import *


def eficient_SBSR(A, LoRs, N, k, knots, lambda0, dmax, options={'maxiter': 2000, 'disp': True}, tol=1e-2):
    '''
    input:
    - A: initial coefficients, usually np.zeros((3, N))
    - LoRs: detected LoRs, shape: (-1, 8)
    - N: number of basis functions
    - k: degree of the spline
    - knots: knots
    
    output
    - optimized A
    '''
    if len(knots) != N + k + 1:
        raise ValueError('knots, N, k are not valid')
    
    sorted_indices = LoRs[:, 3].argsort()
    LoRs = LoRs[sorted_indices]
    
    P1, v, times = get_P1_v_times(LoRs)
    calculateLoss = CalculateLoss(k, knots, times, P1, v, dmax, lambda0)
    result = minimize(lambda A: calculateLoss.get_val_and_grad(A, 1.0), A, method='BFGS', jac=True, options=options, tol=tol)
    
    return result


