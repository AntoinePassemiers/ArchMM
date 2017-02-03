# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp

from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport memcpy
from cython cimport view


ctypedef double bound_t
ctypedef float dom_t
ctypedef double crisp_t
ctypedef int fuzzy_t
ctypedef dom_t[::view.strided, ::1] dom_matrix_dec_t

cdef struct MembershipFunction:
    bound_t lower_bound
    bound_t upper_bound
    dom_t (*base_function)(crisp_t input_value)

cdef dom_matrix_dec_t randomDOMMatrix(size_t n_rows, size_t n_columns)
cdef dom_t computeDOM(MembershipFunction f, crisp_t input_value)