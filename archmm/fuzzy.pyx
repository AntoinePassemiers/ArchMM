# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp

from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport memcpy
from cython cimport view


cdef dom_matrix_dec_t randomDOMMatrix(size_t n_rows, size_t n_columns):
    cdef cnp.ndarray Uarr = np.ascontiguousarray(np.random.rand(n_rows, n_columns))
    Uarr /= Uarr.sum(axis = 1)[:, np.newaxis]
    cdef cnp.double_t[:, :] Ub = Uarr[:, :]
    cdef size_t total_size = n_columns * n_rows * sizeof(dom_t)
    cdef dom_matrix_dec_t U = <dom_t[:n_rows, :n_columns]>malloc(total_size)
    cdef size_t i, j
    for i in range(n_rows):
        for j in range(n_columns):
            U[i, j] = <dom_t>Ub[i, j]
    return U

cdef dom_t computeDOM(MembershipFunction f, crisp_t input_value):
    if input_value < f.lower_bound or input_value > f.upper_bound:
        return 0.0
    else:
        # TODO
        return 0.0
