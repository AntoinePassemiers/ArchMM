# -*- coding: utf-8 -*-
# distutils: language=c

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
    double

ctypedef cnp.double_t datasample_t

cdef inline cnp.double_t[:] inplace_add(cnp.double_t[:] A, cnp.double_t[:] B) nogil

cdef inline cnp.double_t euclidean_distance(cnp.double_t[:] A, cnp.double_t[:] B) nogil

cdef inline double dabs(double value) nogil