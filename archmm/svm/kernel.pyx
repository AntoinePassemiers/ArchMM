# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport memcpy
from cython cimport view

def linear_kernel(X, Y):
    return np.inner(X, Y)

def polynomial_kernel(X, Y, degree = 1, sigma = 1.0, use_bias = False):
    if use_bias:
        return (1.0 + np.inner(X, Y) / sigma) ** degree
    else:
        return (np.inner(X, Y) / sigma) ** degree