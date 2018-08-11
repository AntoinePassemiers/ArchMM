# -*- coding: utf-8 -*-
# math.pyx
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp

from libc.stdlib cimport *
from libc.stdio cimport *
cimport libc.math


cdef cnp.ndarray sample_gaussian(cnp.ndarray mu, cnp.ndarray inv_sigma, int n_samples):
    n_features = mu.shape[0]
    r = np.random.randn(n_samples, n_features)
    return np.dot(r, inv_sigma) + mu


cdef cnp.double_t[:] inplace_add(cnp.double_t[:] A, cnp.double_t[:] B) nogil:
    for i in range(A.shape[0]):
        A[i] = A[i] + B[i]
    return A

cdef cnp.double_t euclidean_distance(cnp.double_t[:] A, cnp.double_t[:] B) nogil:
    return libc.math.sqrt(squared_euclidean_distance(A, B))


cdef cnp.double_t squared_euclidean_distance(cnp.double_t[:] A, cnp.double_t[:] B) nogil:
    cdef size_t i
    cdef cnp.double_t result = 0.0
    for i in range(A.shape[0]):
        result += (A[i] - B[i]) ** 2
    return result