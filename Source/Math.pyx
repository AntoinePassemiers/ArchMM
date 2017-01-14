# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as cnp
import cython
from libc.stdlib cimport *
from libc.stdio cimport *
cimport libc.math

ctypedef fused primitive_t:
    cnp.float_t
    cnp.double_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

ctypedef cnp.double_t datasample_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cnp.double_t[:] inplace_add(cnp.double_t[:] A, cnp.double_t[:] B) nogil:
    for i in range(A.shape[0]):
        A[i] = A[i] + B[i]
    return A

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cnp.double_t euclidean_distance(cnp.double_t[:] A, cnp.double_t[:] B) nogil:
    cdef size_t i
    cdef cnp.double_t result = 0.0
    for i in range(A.shape[0]):
        result += (A[i] - B[i]) ** 2
    return libc.math.sqrt(result)


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