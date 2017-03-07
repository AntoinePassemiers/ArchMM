# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp

import cython

cimport libc.math
from libc.stdlib cimport *
from libc.stdio cimport *


ctypedef cnp.double_t sample_vector_t
ctypedef cnp.double_t dev_alloc_data_t
ctypedef cnp.double_t[:] dev_alloc_vector_t

ctypedef struct gaussianSample2d:
    cnp.float_t X
    cnp.float_t Y

ctypedef cnp.double_t datasample_t
cdef inline cnp.double_t[:] inplace_add(cnp.double_t[:] A, cnp.double_t[:] B) nogil
cdef inline cnp.double_t euclidean_distance(cnp.double_t[:] A, cnp.double_t[:] B) nogil
cdef inline double dabs(double value) nogil
cdef inline cnp.float_t univariateBoxMullerMethod() nogil
cdef inline gaussianSample2d* bivariateBoxMullerMethod() nogil
cdef inline cnp.float_t univariateMarsagliaPolarMethod() nogil
cdef inline gaussianSample2d* bivariateMarsagliaPolarMethod() nogil
cdef inline cnp.double_t cMahalanobisDistance(cnp.double_t[:] X, cnp.double_t[:] mu, 
                                              cnp.double_t[:, :] inv_sigma) nogil