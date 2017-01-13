# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as cnp
import cython
from libc.stdlib cimport *
from libc.stdio cimport * 

ctypedef fused primitive_t:
    cnp.float_t
    cnp.double_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cnp.double_t* cy_add(cnp.double_t[:] A, cnp.double_t[:] B) nogil:
    cdef size_t i
    cdef cnp.double_t* result = <cnp.double_t*>malloc(A.shape[0] * sizeof(cnp.double_t))
    for i in range(A.shape[0]):
        result[i] = A[i] + B[i]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cnp.double_t* cy_subtract(cnp.double_t[:] A, cnp.double_t[:] B) nogil:
    cdef size_t i
    cdef cnp.double_t* result = <cnp.double_t*>malloc(A.shape[0] * sizeof(cnp.double_t))
    for i in range(A.shape[0]):
        result[i] = A[i] - B[i]
    return result


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