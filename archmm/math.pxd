# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp


cdef cnp.ndarray sample_gaussian(cnp.ndarray mu, cnp.ndarray inv_sigma, int n_samples)

cdef cnp.double_t[:] inplace_add(cnp.double_t[:] A, cnp.double_t[:] B) nogil
cdef cnp.double_t euclidean_distance(cnp.double_t[:] A, cnp.double_t[:] B) nogil
cdef cnp.double_t squared_euclidean_distance(cnp.double_t[:] A, cnp.double_t[:] B) nogil