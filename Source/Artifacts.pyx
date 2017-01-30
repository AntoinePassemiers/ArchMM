# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport * 

""" TODO
- Detect drop-outs sequences
https://www.quora.com/How-can-I-estimate-the-parameters-of-a-discrete-time-HMM-when-some-observations-are-missing
- Detect outliers
"""

DEF DEFAULT_MISSING_VALUE = -87.89126015

cpdef cnp.ndarray getMissingValuesIndexes(cnp.ndarray data, double missing_value):
    return np.where(data == missing_value)[0]

NUMPY_NAN = 0.0
NUMPY_INF = np.nan_to_num(np.inf)

def nan_to_num(arr):
    """ Inplace NaN conversion in numpy arrays """
    """ TODO : replace np.isnan by a pure C loop (np.isnan is wasting memory) """
    arr[np.isnan(arr)] = NUMPY_NAN
    return arr

cpdef cnp.ndarray hasMissingValues(data, missing_value, n_datadim = 2):
    cdef unsigned int ndim = len(data.shape)
    assert(ndim > 1)
    cdef unsigned int N = data.shape[0]
    cdef unsigned int T = data.shape[1]
    cdef cnp.ndarray has_mv
    cdef Py_ssize_t i, t
    if n_datadim == 1:
        has_mv = np.zeros(N, dtype = np.bool)
        if missing_value is np.nan:
            for i in range(N):
                if np.isnan(data[i]).any():
                    has_mv[i] = True
        elif missing_value is np.inf:
            for i in range(N):
                if np.isinf(data[i]).any():
                    has_mv[i] = True
        else:
            for i in range(N):
                if missing_value in data[i]:
                    has_mv[i] = True
    elif ndim > 2:
        has_mv = np.zeros((N, T), dtype = np.bool)
        if missing_value is np.nan:
            for i in range(N):
                for t in range(T):
                    if np.isnan(data[i, t]).any():
                        has_mv[i, t] = True
        elif missing_value is np.inf:
            for i in range(N):
                for t in range(T):
                    if np.isinf(data[i, t]).any():
                        has_mv[i, t] = True
        else:
            for i in range(N):
                for t in range(T):
                    if missing_value in data[i, t]:
                        has_mv[i, t] = True
    else:
        has_mv = np.zeros((N, T), dtype = np.bool)
        for i in range(N):
            for t in range(T):
                if missing_value == data[i, t]:
                    has_mv[i, t] = True
    return has_mv