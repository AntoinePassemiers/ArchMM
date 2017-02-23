# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport *
cimport libc.math


ctypedef cnp.double_t[:] np_vector_data_t

cdef inline cnp.double_t[:] inplace_add(cnp.double_t[:] A, cnp.double_t[:] B) nogil:
    for i in range(A.shape[0]):
        A[i] = A[i] + B[i]
    return A

cdef inline cnp.double_t euclidean_distance(cnp.double_t[:] A, cnp.double_t[:] B) nogil:
    cdef size_t i
    cdef cnp.double_t result = 0.0
    for i in range(A.shape[0]):
        result += (A[i] - B[i]) ** 2
    return libc.math.sqrt(result)

cdef inline double dabs(double value) nogil:
    return -value if value < 0 else value

def stableInvSigma(sigma):
    sigma = np.nan_to_num(sigma)
    singular = True
    mcv = 0.00001
    while singular:
        try:
            inv_sigma = np.array(np.linalg.inv(sigma), dtype = np.double)
            singular = False
        except np.linalg.LinAlgError:
            sigma += np.eye(len(sigma), dtype = np.double) * mcv # TODO
            mcv *= 10
    return inv_sigma