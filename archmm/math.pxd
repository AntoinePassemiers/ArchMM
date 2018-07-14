# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp

import cython

cimport libc.math
from libc.stdlib cimport *
from libc.stdio cimport *


cdef cnp.ndarray sample_gaussian(cnp.ndarray mu, cnp.ndarray inv_sigma, int n_samples)

cdef cnp.double_t[:] inplace_add(cnp.double_t[:] A, cnp.double_t[:] B) nogil
cdef cnp.double_t euclidean_distance(cnp.double_t[:] A, cnp.double_t[:] B) nogil