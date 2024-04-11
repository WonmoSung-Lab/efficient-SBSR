import numpy as np
import scipy


def categorize_list(times, ranges):
    '''
    categorize the times for 'get_precal_basis' function
    '''
    categories = [0]
    i = 1
    for time_index in range(times.size):
        for j in range(i, len(ranges)-1):
            if times[time_index] < ranges[j]:
                break
            categories.append(time_index)
            i += 1
    categories.extend([time_index + 1] * (len(ranges)-i))
    return np.array(categories)

def get_precal_basis(k, knots, times):
    '''
    input:
      - k: degree of spline
      - times: array of timestamps for each detected LoR
      - knots
      
    output: matrix B defined as B_{nj} = b_{n-1,3}(times_j)
    '''
    N = len(knots) - k - 1
    categories = categorize_list(times, knots[k:-k])
    basis_precalculated = np.zeros((N, times.size))

    for i in range(N-k):
        temp = []
        for j in range(categories[i], categories[i+1]):
            temp.append(scipy.interpolate._bspl.evaluate_all_bspl(knots, k, times[j], i + k))
        basis_precalculated[i:i+k+1, categories[i]:categories[i+1]] = np.array(temp).T

    return basis_precalculated
    
class CalculateM:
    '''
    forward: Calculate M, a matrix which each column representing the estimated location at the detected LoR's timestamps
    
    backwards: Calculate dI/dA from dI/dM using backpropagation
    '''
    def __init__(self, precal_basis):
        self.precal_basis = precal_basis
        self.A = None

    def forward(self, A):
        self.A = A
        out = np.dot(self.A, self.precal_basis)
        return out

    def backward(self, dout):
        dA = np.dot(dout, self.precal_basis.T)
        return dA
    
class CalculateMP:
    '''
    forward: Calculate MP, a matrix which each column representing a vector from the estimated location at the detected LoR's timestamps to the location of one of the detected detectors
    
    backwards: Calculate dI/dM from dI/dMP using backpropagation
    '''
    def __init__(self, P1):
        self.P1 = P1
        self.M = None

    def forward(self, M):
        self.M = M
        out = self.P1 - self.M
        return out

    def backward(self, dout):
        dM = - dout
        return dM
    
class CalculateDSquared:
    '''
    forward: Calculate D^2, a matrix which each column representing the square distance from the estimated location at the detected LoR's timestamps to the LoR
    
    backwards: Calculate dI/dMP from dI/dD^2 using backpropagation
    '''
    def __init__(self, v):
        self.v = v
        self.MP = None
        self.MP_shape = None
        self.MP_v = None

    def forward(self, MP):
        self.MP = MP
        self.MP_shape = MP.shape
        self.MP_v = (self.MP * self.v).sum(axis=0)
        out = (self.MP ** 2).sum(axis=0) - (self.MP_v ** 2)
        return out

    def backward(self, dout):
        k1 = (2 * (- dout) * self.MP_v)
        k2 = np.tile(k1, (3, 1))
        dMP = (k2 * self.v) + (2 * dout * self.MP)
        return dMP
    
class CalculateI:
    '''
    forward: Calculate I, the restricted mean squared distances
    
    backwards: Calculate dI/dD^2 from dI/dI(= 1) using backpropagation
    '''
    def __init__(self, dmax):
        self.dmax = dmax
        self.mask = None
        self.times_count = None
        
    def forward(self, d_squared):
        self.times_count = d_squared.size
        self.mask = (d_squared > (self.dmax ** 2))
        out = d_squared.copy()
        out[self.mask] = (self.dmax ** 2)
        return out.sum() / self.times_count

    def backward(self, dout):
        dout = np.full((self.times_count), dout) / self.times_count
        dout[self.mask] = 0
        dDSquared = dout
        return dDSquared
    
class CalculateR:
    '''
    forward: Calculate R, the mean squared difference between consecutive columns of A
    
    backwards: Calculate dR/dA from dR/dR(= 1) using backpropagation
    '''
    def __init__(self, lambda0):
        self.lambda0 = lambda0
        self.a = None
        self.a_shape = None

    def forward(self, a):
        self.a = a
        self.a_shape = a.shape
        self.extract = a[:, 1:] - a[:, 0:-1]
        out = (self.extract ** 2).sum() / self.a_shape[1]
        return self.lambda0 * out

    def backward(self, dout):
        dSquares = np.full((self.a_shape[0], self.a_shape[1] - 1), (dout * self.lambda0 / self.a_shape[1]))
        dExtract = 2 * dSquares * self.extract
        da = np.hstack((np.zeros((self.a_shape[0], 1)), dExtract)) - np.hstack((dExtract, np.zeros((self.a_shape[0], 1))))
        return da
    
class CalculateLoss:
    def __init__(self, k, knots, times, P1, v, dmax, lambda0):
        '''
        times, P1, v needs to be sorted in increasing time order
        k is usually 3 (cubic spline)
        knots needs padding
        '''
        self.calculateM = CalculateM(get_precal_basis(k, knots, times))
        self.calculateMP = CalculateMP(P1)
        self.calculateDSquared = CalculateDSquared(v)
        self.calculateI = CalculateI(dmax)
        self.calculateR = CalculateR(lambda0)
        
        self.val = None
        self.loss_values = []

    def forward(self, A):
        M = self.calculateM.forward(A)
        MP = self.calculateMP.forward(M)
        DSquared = self.calculateDSquared.forward(MP)
        I = self.calculateI.forward(DSquared)
        
        R = self.calculateR.forward(A)
        
        return I + R

    def backward(self, dout):
        dDSquared = self.calculateI.backward(dout)
        dMP = self.calculateDSquared.backward(dDSquared)
        dM = self.calculateMP.backward(dMP)
        da1 = self.calculateM.backward(dM)
        da2 = self.calculateR.backward(dout)
        return da1 + da2
    
    def get_val_and_grad(self, A, dout):
        A = A.reshape((3, -1))
        self.val = self.forward(A)
        grad = self.backward(dout).ravel()
        return self.val, grad
    
    # function for callback
    def callback(self):
        self.loss_values.append(self.val)