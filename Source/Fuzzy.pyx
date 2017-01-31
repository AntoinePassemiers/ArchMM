# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport *


ctypedef double bound_t
ctypedef float dom_t
ctypedef double crisp_t
ctypedef int fuzzy_t

cdef struct MembershipFunction:
	bound_t lower_bound
	bound_t upper_bound
	dom_t (*base_function)(crisp_t input_value)

cdef dom_t computeDOM(MembershipFunction f, crisp_t input_value):
	if input_value < f.lower_bound or input_value > f.upper_bound:
		return 0.0
	else:
		# TODO
		return 0.0