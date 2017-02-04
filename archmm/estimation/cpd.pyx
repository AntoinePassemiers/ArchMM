# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
cnp.import_array()

import multiprocessing
from cython.parallel import parallel, prange, threadid

from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport memset

from archmm.math import *
from archmm.math cimport *
from archmm.queue cimport *
from archmm.utils cimport *


cdef double NUMPY_INF_VALUE = np.nan_to_num(np.inf)

cpdef unsigned int POLYNOMIAL_APRX = 1
cpdef unsigned int WAVELET_APRX = 2
cpdef unsigned int FOURIER_APRX = 3

cpdef unsigned int SUM_OF_SQUARES_COST = 4
cpdef unsigned int MAHALANOBIS_DISTANCE_COST = 5

cpdef unsigned int KERNEL_RADIAL = 100
cpdef unsigned int KERNEL_TRICUBIC = 101


def epolyfit(arr, degree, **kwargs):
    N = arr.shape[0]
    ndim = arr.shape[1]
    if N <= degree:
        reg = (np.zeros(1), np.zeros(ndim), 0, np.zeros(1), 0)
    else:
        X = np.arange(N)
        reg = np.polyfit(X, arr, degree, **kwargs)
    return reg

cdef double SumOfSquaresCost(cnp.double_t[:] vector):
    """ Sum of squares of the vector components """
    cdef Py_ssize_t i
    cdef double total = 0.0
    for i in range(len(vector)):
        total += vector[i] ** 2
    return total
    
cdef double MahalanobisCost(cnp.double_t[:] vector, cnp.ndarray mu, cnp.ndarray inv_sigma):
    """ Mahalanobis distance between the vector and the mean of the whole sequence """
    cdef Py_ssize_t n = len(vector)
    if n == 0:
        return 0
    cdef cnp.ndarray delta = np.sqrt(np.nan_to_num(np.asarray(vector))) - mu
    cost = np.dot(np.dot(delta, inv_sigma), delta)
    return <double>cost

cdef struct fog_t:
    cnp.double_t value
    Py_ssize_t begin
    Py_ssize_t end

cdef class MB:
    def __cinit__(self):
        pass

cdef class GTD(MB):
    cdef Py_ssize_t window_size
    cdef unsigned int cost_func
    cdef cnp.int16_t[:] keypoints
    cdef fog_t* fogs

    def __cinit__(self, cost_func = MAHALANOBIS_DISTANCE_COST,
                  window_size = 15, n_keypoints = 50):
        self.window_size = window_size
        self.cost_func = cost_func
        self.keypoints = np.empty(n_keypoints, dtype = np.int16)

    cpdef detectPoints(self, sequence):
        ensure_PyObject_Buffer(sequence)
        cdef Py_ssize_t n_threads = multiprocessing.cpu_count()
        cdef Py_ssize_t i, j, k, t, T = len(sequence)
        cdef Py_ssize_t best_i, best_j
        cdef Py_ssize_t N = self.window_size
        cdef cnp.double_t[:] beta = np.empty(N, dtype = np.double)
        cdef cnp.double_t[:] C = np.empty(n_threads, dtype = np.double)
        cdef double fog, C_prime
        cdef cnp.double_t[:, :] signal_buffer = np.asarray(sequence)
        self.fogs = <fog_t*>malloc((T - N) * sizeof(fog_t))
        with nogil:
            for t in prange(0, T - N):
                k = threadid()
                memset(<void*>&beta[0], 0x00, N * sizeof(cnp.double_t))
                fog = -NUMPY_INF_VALUE
                for j in range(N - 2, 1, -1):
                    C[k] = 0.0
                    for i in range(j - 1, 0, -1):
                        beta[i] += self.getCost(signal_buffer[t + i], signal_buffer[t + j])
                        C[k] += beta[i]
                        C_prime = C[k] / ((j - i) * (N - j))
                        if C_prime > fog:
                            fog = C_prime
                            best_i, best_j = i + t, j + t
                printf("%i, %i, %i, %d\n", k, best_j, best_i, fog)
                self.fogs[t].begin = best_i
                self.fogs[t].end = best_j
                self.fogs[t].value = fog
        
    cpdef cnp.int_t[:] getKeypoints(self):
        pass
    
    cdef inline cnp.double_t getCost(self, cnp.double_t[:] A, cnp.double_t[:] B) nogil except? -1:
        cdef cnp.double_t cost
        if self.cost_func == MAHALANOBIS_DISTANCE_COST:
            """
            cost = MahalanobisCost(costs_A, self.mu, self.inv_sigma) + \
                MahalanobisCost(costs_B, self.mu, self.inv_sigma)
            """
            pass # TODO
        elif self.cost_func == SUM_OF_SQUARES_COST:
            cost = euclidean_distance(A, B)
        else:
            with gil:
                raise NotImplementedError()
        return cost


cdef inline cnp.double_t computeTStat(cnp.double_t s_tot, cnp.double_t s_r, cnp.double_t s_l,
                                      Py_ssize_t N, Py_ssize_t i, Py_ssize_t j) nogil:
    cdef cnp.double_t num = s_r / (N - j) - s_l / (j - i)
    cdef cnp.double_t A = (s_tot ** 2 - s_l ** 2 / (j - i) - s_r ** 2 / (N - j)) / (N - i - 2)
    cdef cnp.double_t B = (N - i) / ((j - i) * (N - j))
    return A / libc.math.sqrt(A * B)

cdef class TSTAT(MB):
    cdef Py_ssize_t window_size
    cdef unsigned int cost_func
    cdef cnp.int16_t[:] keypoints
    cdef fog_t* fogs

    def __cinit__(self, kernel = KERNEL_RADIAL, window_size = 15, n_keypoints = 50):
        self.kernel = kernel
        self.window_size = window_size
        self.keypoints = np.empty(n_keypoints, dtype = np.int16)

    cpdef detectPoints(self, signal):
        ensure_PyObject_Buffer(signal)
        cdef Py_ssize_t n_threads = multiprocessing.cpu_count()
        cdef Py_ssize_t i, j, k, t, T = len(signal)
        cdef Py_ssize_t r = signal.shape[1]
        cdef Py_ssize_t best_i, best_j
        cdef Py_ssize_t N = self.window_size
        cdef double fog, tstat
        cdef cnp.double_t[:] S_tot, S_tot_2, S_r, S_l
        cdef cnp.double_t[:, :] signal_buffer = np.asarray(signal)
        self.fogs = <fog_t*>malloc((T - N) * sizeof(fog_t))
        """
        with nogil:
            for t in prange(0, T - N):
                k = threadid()
                fog = -NUMPY_INF_VALUE
                S_tot = signal_buffer[t + N - 1]
                S_tot_2 = signal_buffer[t + N - 1] ** 2
                for i in range(N - 2, 0, -1):
                    S_tot = inplace_add(S_tot, signal_buffer[t + i])
                    S_tot_2 = inplace_add(S_tot_2, signal_buffer[t + i] ** 2)
                    memset(<void*>&S_r[0], 0x00, r * sizeof(cnp.double_t))
                    for j in range(N - 1, i - 1, -1):
                        S_r = inplace_add(S_r, signal_buffer[t + j])
                        S_l = S_tot - S_r
                        tstat = computeTStat(s_tot, s_r, s_l, N, i, j)
                        if tstat > fog:
                            fog = tstat
                            best_i, best_j = i + t, j + t
                printf("%i, %i, %i, %d\n", k, best_j, best_i, fog)
                self.fogs[t].begin = best_i
                self.fogs[t].end = best_j
                self.fogs[t].value = fog
        """

    cpdef cnp.int_t[:] getKeypoints(self):
        pass

cdef class BatchCPD:
    """Implementation of the batch Change Point Detection Algorithm.
    This algorithm is designed for getting the positions of the points where most of the
    changes occur in the signal. These points are called change points.
    The input signal is processed according to a divide-and-conquer procedure.
    
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
                  cost_func = MAHALANOBIS_DISTANCE_COST, max_n_keypoints = 50, window_padding = 1, 
                  n_keypoints = None):
        if not (POLYNOMIAL_APRX <= aprx_func <= FOURIER_APRX):
            raise NotImplementedError(str(aprx_func) + " approximation function is not supported.")
        self.aprx_func = aprx_func
        self.aprx_degree = aprx_degree
        self.stability_threshold = threshold
        self.cost_func = cost_func
        self.max_n_keypoints = max_n_keypoints if not n_keypoints else n_keypoints # TODO
        self.n_keypoints = 0
        self.n_potential_points = 0
        assert(window_padding > 0)
        self.window_padding = window_padding
        self.keypoints = np.empty(self.max_n_keypoints + 1, dtype = np.int)
        
    def __dealloc__(self):
        free(self.costs)

    cpdef detectPoints(self, signal, mu, sigma):
        """ Initializes the parameters, the data structures, and calls the function
         evaluateSegment.
         
        Parameters
        ----------
        signal : array containing the input signal
        """
        assert(len(signal.shape) == 2)
        self.signal = signal
        self.n = signal.shape[0]
        self.n_dim = signal.shape[1]
        self.mu = mu
        self.inv_sigma = stableInvSigma(sigma)
        self.potential_points = <Py_ssize_t*>malloc(2 * self.n * sizeof(Py_ssize_t))
        self.costs = <double*>malloc(2 * self.n * sizeof(double))
        if self.costs == NULL:
            printf("Memory error.")
            exit(EXIT_FAILURE)
        self.evaluateSegment()
        self.keypoints[self.n_keypoints] = self.n
        self.keypoints[self.n_keypoints + 1] = 0
        assert(len(self.keypoints) == self.n_keypoints + 2)
        self.keypoints = np.sort(self.keypoints[:self.n_keypoints + 2])
        print(list(self.keypoints))
        for i in range(len(self.keypoints) - 1):
            assert(self.keypoints[i] != self.keypoints[i + 1])
        
    cpdef cnp.int_t[:] getKeypoints(self):
        return self.keypoints
    
    cdef double getCost(self, cnp.ndarray costs_A, cnp.ndarray costs_B) except? -1:
        cdef double cost
        if self.cost_func == MAHALANOBIS_DISTANCE_COST:
            # TODO : remplacer aprx_A[1] par aprx_B[1] par les approximations de la régression
            cost = MahalanobisCost(costs_A, self.mu, self.inv_sigma) + \
                MahalanobisCost(costs_B, self.mu, self.inv_sigma)
        elif self.cost_func == SUM_OF_SQUARES_COST:
            cost = SumOfSquaresCost(costs_A) + SumOfSquaresCost(costs_B)
        else:
            raise NotImplementedError()
        return cost
    
    cdef void evaluateSegment(self):
        """ Function which fits the subset between [[begin]] end [[end]].
        The cost of the regression is used as a criterion to find the best split point 
        for the current subset. The next two subsets are then created from the split that 
        minimizes the cost function, and are processed according to a breadth-first search.
         
        Parameters
        ----------
        begin : index of the subset's first sample
        end : index of the subset's last sample
        previous_cost : cost function of the previous step
                        The convergence criterion is fulfilled when 
                        ([[previous_cost]] - cost) / [[previous_cost]] is lower than
                        a given threshold.
        """
        cdef Queue* queue = newQueue()
        cdef Iteration* current_iter = <Iteration*>malloc(sizeof(Iteration))
        current_iter.begin = 0
        current_iter.end = self.n
        current_iter.previous_cost = NUMPY_INF_VALUE
        cdef cnp.double_t[:] aprx
        cdef Py_ssize_t begin, end
        cdef double previous_cost, cost = 0
        cdef size_t slice_size
        cdef Py_ssize_t mid, best_mid
        cdef double lowest_cost
        enqueue(queue, current_iter)
        while (not (self.n_keypoints == self.max_n_keypoints - 1)) and (not isQueueEmpty(queue)):
            current_iter = dequeue(queue)
            begin = current_iter.begin
            end = current_iter.end
            previous_cost = current_iter.previous_cost
            free(current_iter)
            assert(begin < end)
            slice_size = end - begin
            best_mid = begin + slice_size / 2
            lowest_cost = NUMPY_INF_VALUE
            # assert(slice_size > 2 * self.window_padding)
            for mid in range(begin + self.window_padding, end - self.window_padding):
                cost = 0
                if self.aprx_func == POLYNOMIAL_APRX:
                    aprx_A = epolyfit(self.signal[begin:mid], self.aprx_degree, full = True)
                    aprx_B = epolyfit(self.signal[mid:end], self.aprx_degree, full = True)
                    cost = self.getCost(aprx_A[1], aprx_B[1])
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_mid = mid
            self.potential_points[self.n_potential_points] = best_mid
            self.costs[self.n_potential_points] = lowest_cost
            self.n_potential_points += 1
            # TODO : recursive call only for the lowest weighted cost
            # TODO : check potential_points to determine the new keypoint
            # if not (self.n_keypoints == self.max_n_keypoints - 1):
            self.keypoints[self.n_keypoints] = best_mid
            self.n_keypoints += 1
            # TODO : if (previous_cost - cost) / previous_cost >= self.stability_threshold:
            if not (self.n_keypoints == self.max_n_keypoints - 1):
                if best_mid - begin > 2:
                    current_iter = <Iteration*>malloc(sizeof(Iteration))
                    current_iter.begin = begin + 1
                    current_iter.end = best_mid
                    current_iter.previous_cost = cost
                    enqueue(queue, current_iter)
                if end - best_mid > 2:
                    current_iter = <Iteration*>malloc(sizeof(Iteration))
                    current_iter.begin = best_mid
                    current_iter.end = end - 1
                    current_iter.previous_cost = cost
                    enqueue(queue, current_iter)
                # self.evaluateSegment(begin + 1, best_mid, cost)
                # self.evaluateSegment(best_mid, end - 1, cost)
        free(queue)


        