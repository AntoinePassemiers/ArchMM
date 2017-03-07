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

from archmm.utils cimport *


cdef cnp.float_t c_PI = np.pi

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

cdef inline cnp.float_t univariateBoxMullerMethod() nogil:
    cdef cnp.float_t U = cRand()
    cdef cnp.float_t V = cRand()
    return libc.math.sqrt(-2.0 * libc.math.log(U)) * libc.math.cos(2.0 * c_PI * V)

cdef inline gaussianSample2d* bivariateBoxMullerMethod() nogil:
    cdef cnp.float_t U = cRand()
    cdef cnp.float_t V = cRand()
    cdef gaussianSample2d* sample = <gaussianSample2d*>malloc(sizeof(gaussianSample2d))
    sample.X = libc.math.sqrt(-2.0 * libc.math.log(U)) * libc.math.cos(2.0 * c_PI * V)
    sample.Y = libc.math.sqrt(-2.0 * libc.math.log(V)) * libc.math.cos(2.0 * c_PI * U)
    return sample

cdef inline cnp.float_t univariateMarsagliaPolarMethod() nogil:
    cdef cnp.float_t U = cRand()
    cdef cnp.float_t V = cRand()
    cdef cnp.float_t S = U * U + V * V
    return U * libc.math.sqrt(-2.0 * libc.math.log(S) / S)

cdef inline gaussianSample2d* bivariateMarsagliaPolarMethod() nogil:
    cdef cnp.float_t U = cRand()
    cdef cnp.float_t V = cRand()
    cdef cnp.float_t S = U * U + V * V
    cdef gaussianSample2d* sample = <gaussianSample2d*>malloc(sizeof(gaussianSample2d))
    sample.X = U * libc.math.sqrt(-2.0 * libc.math.log(S) / S)
    sample.Y = V * libc.math.sqrt(-2.0 * libc.math.log(S) / S)
    return sample

cdef inline cnp.double_t cMahalanobisDistance(cnp.double_t[:] X, cnp.double_t[:] mu, 
                                              cnp.double_t[:, :] inv_sigma) nogil:
    dev_alloc_vector_t diff = <dev_alloc_data_t[:X.shape[0]]>malloc(X.shape[0] * sizeof(dev_alloc_data_t))
    cdef size_t i, j
    for i in range(X.shape[0]):
        diff[i] = X[i] - mu[i]
    # TODO : quadratic multiplication : diff.T * inv_sigma * diff


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