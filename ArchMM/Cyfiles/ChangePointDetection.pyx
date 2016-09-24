# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport * 
from cython.parallel import parallel, prange

include "Math.pyx"

#http://iopscience.iop.org/article/10.1088/1742-6596/364/1/012031/pdf

cdef double NUMPY_INF_VALUE = np.nan_to_num(np.inf)

cpdef unsigned int POLYNOMIAL_APRX = 1
cpdef unsigned int WAVELET_APRX = 2
cpdef unsigned int FOURIER_APRX = 3

cpdef unsigned int SUM_OF_SQUARES_COST = 4
cpdef unsigned int MAHALANOBIS_DISTANCE_COST = 5


cdef class BatchCPD:
    """Implementation of the batch Change Point Detection Algorithm.
    This algorithm is designed for getting the positions of the points where most of the
    changes occur in the signal. These points are called change points.
    The input signal is processed according to a recursive divide-and-conquer procedure.
    
    Parameters
    ----------
    aprx_func : type of function used for fitting the input data
                [POLYNOMIAL_APRX] Polynomial regression
                [WAVELET_APRX] Wavelet transform
                [FOURIER_APRX] Fourier transform
    aprx_degree : if aprx_func == POLYNOMIAL_APRX, aprx_degree is the degree of the
                  corresponding polynomial
                  if aprx_func == WAVELET_APRX, aprx_degree is the number of coefficients
                  in the corresponding wavelet transform
                  if aprx_func == FOURIER_APRX, aprx_degree is the number of coefficients
                  in the corresponding Fourier transform
    threshold : minimum decrease rate for the cost function
                Under this value, we consider that the algorithm has converged.
    cost_func : type of function used for computing the cost of the algorithm
                [SUM_OF_SQUARES_COST] Sum of square function
                [MAHALANOBIS_DISTANCE_COST] Mahalanobis distance
    n_key_points : number of optimal change points we want to get
    max_n_keypoints : maximum number of change points
                      If max_n_points is provided but n_key_points is not provided,
                      the number will be a power of 2.
    window_padding : minimum number of frames to execute the approximation function on
        
    Attributes
    ----------
    keypoints : indexes of the change points
    costs : array containing the cost of each subdivision of the signal
            A subdivision is a region between two change points.
    mu : mean of the input signal
    inv_sigma : inverse of the variance-covariance matrix of the input signal
    """
    
    cdef size_t window_padding 
    cdef unsigned int aprx_func, aprx_degree
    cdef float stability_threshold
    cdef unsigned int cost_func
    cdef cnp.int_t[:] keypoints
    cdef size_t n_keypoints, max_n_keypoints
    cdef cnp.ndarray signal
    cdef Py_ssize_t n, n_dim
    cdef double* costs
    cdef Py_ssize_t* potential_points
    cdef Py_ssize_t n_potential_points
    cdef cnp.double_t[:] mu
    cdef cnp.double_t[:, :] inv_sigma
    
    def __cinit__(self, aprx_func = POLYNOMIAL_APRX, aprx_degree = 3, threshold = 0.01,
                  cost_func = SUM_OF_SQUARES_COST, max_n_keypoints = 50, window_padding = 7, n_keypoints = None):
        if not (POLYNOMIAL_APRX <= aprx_func <= FOURIER_APRX):
            raise NotImplementedError(str(aprx_func) + " approximation function is not supported.")
        self.aprx_func = aprx_func
        self.aprx_degree = aprx_degree
        self.stability_threshold = threshold
        self.cost_func = cost_func
        self.max_n_keypoints = max_n_keypoints if not n_keypoints else n_keypoints # TODO
        self.n_keypoints = 0
        self.n_potential_points = 0
        self.window_padding = window_padding
        self.keypoints = np.empty(self.max_n_keypoints + 1, dtype = np.int)
        self.keypoints[0] = 0
        
    def __dealloc__(self):
        free(self.costs)

    cpdef detectPoints(self, signal):
        """ Initializes the parameters, the data structures, and calls the recursive function
         evaluateSegment.
         
        Parameters
        ----------
        signal : array containing the input signal
        """
        assert(len(signal.shape) == 2)
        self.signal = signal
        self.n = signal.shape[0]
        self.n_dim = signal.shape[1]
        self.mu = np.mean(signal, axis = 0) ** 2
        self.inv_sigma = stableInvSigma(np.cov(signal.T))
        self.potential_points = <Py_ssize_t*>malloc(2 * self.n * sizeof(Py_ssize_t))
        self.costs = <double*>malloc(2 * self.n * sizeof(double))
        if self.costs == NULL:
            printf("Memory error.")
            exit(EXIT_FAILURE)
        self.evaluateSegment(0, self.n, NUMPY_INF_VALUE)
        print(len(self.keypoints), self.n_keypoints + 2)
        assert(len(self.keypoints) == self.n_keypoints + 2)
        self.keypoints[self.n_keypoints] = self.n
        self.keypoints = np.sort(self.keypoints[:self.n_keypoints+1])
        print(list(self.keypoints[:]))
        for i in range(len(self.keypoints) - 1):
            assert(self.keypoints[i] != self.keypoints[i + 1])
        
    cpdef cnp.int_t[:] getKeypoints(self):
        return self.keypoints
    
    cdef double MahalanobisCost(self, cnp.double_t[:] vector):
        """ Mahalanobis distance between the vector and the mean of the signal """
        cdef Py_ssize_t i, n = len(vector)
        cdef cnp.ndarray delta = np.empty(len(vector))
        for i in range(n):
            delta[i] = vector[i] - self.mu[i]
        return <double>np.dot(np.dot(delta, self.inv_sigma), delta)
    
    cdef double SumOfSquaresCost(self, cnp.double_t[:] vector):
        """ Sum of squares of the vector elements """
        cdef Py_ssize_t i
        cdef double total = 0.0
        for i in range(len(vector)):
            total += vector[i] ** 2
        return total
    
    cdef void evaluateSegment(self, Py_ssize_t begin, Py_ssize_t end, double previous_cost):
        """ Recursive function which fits the subset between [[begin]] end [[end]].
        The cost of the regression is used as a criterion to find the best split point 
        for the current subset. The next two subsets are then created from the split that 
        minimizes the cost function, and are processed recursively.
         
        Parameters
        ----------
        begin : index of the subset's first sample
        end : index of the subset's last sample
        previous_cost : cost function of the previous step
                        The convergence criterion is fulfilled when 
                        ([[previous_cost]] - cost) / [[previous_cost]] is lower than
                        a given threshold/ 
        """
        assert(begin < end)
        cdef cnp.double_t[:] aprx
        cdef double cost = 0
        cdef size_t slice_size = end - begin
        cdef Py_ssize_t mid, best_mid = begin + slice_size / 2
        cdef double lowest_cost = NUMPY_INF_VALUE
        # TODO : polyfit for window_padding < 7
        if slice_size > 2 * self.window_padding:
            for mid in range(begin + self.window_padding, end - self.window_padding):
                cost = 0
                if self.aprx_func == POLYNOMIAL_APRX:
                    aprx_A = np.polyfit(np.arange(mid - begin), 
                            self.signal[begin:mid], self.aprx_degree, full = True)
                    aprx_B = np.polyfit(np.arange(end - mid), 
                            self.signal[mid:end], self.aprx_degree, full = True)
                    if self.cost_func == MAHALANOBIS_DISTANCE_COST:
                        # TODO : remplacer aprx_A[1] par aprx_B[1] par les approximations de la régression
                        cost = self.MahalanobisCost(aprx_A[1]) + self.MahalanobisCost(aprx_B[1])
                    elif self.cost_func == SUM_OF_SQUARES_COST:
                        cost = self.SumOfSquaresCost(aprx_A[1]) + self.SumOfSquaresCost(aprx_B[1])
                    else:
                        pass # TODO
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_mid = mid
            self.potential_points[self.n_potential_points] = best_mid
            self.costs[self.n_potential_points] = lowest_cost
            self.n_potential_points += 1
            # TODO : recursive call only for the lowest weighted cost
            # TODO : check potential_points to determine the new keypoint
            self.keypoints[self.n_keypoints] = best_mid
            self.n_keypoints += 1
            if not (self.n_keypoints == self.max_n_keypoints - 1):
                # TODO : if (previous_cost - cost) / previous_cost >= self.stability_threshold:
                self.evaluateSegment(begin + 1, best_mid, cost)
                self.evaluateSegment(best_mid, end - 1, cost)


        