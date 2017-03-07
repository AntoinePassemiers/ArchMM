# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport *

from scipy import stats


DEFAULT_MISSING_VALUE = -87.89126015

NUMPY_NAN = 0.0
NUMPY_INF = np.nan_to_num(np.inf)


class GrubbsTest:
    def __init__(self, N, alpha = 0.01):
        self.N = N
        self.alpha = alpha
        self.crit_value = stats.t.isf(alpha / float(2 * N), N - 2)
    def test(self, signal, replace = True):
        N = len(signal)
        Y_argmin = signal.argmin()
        Y_argmax = signal.argmax()
        Y_argopt = Y_argmax if signal[Y_argmax] > - signal[Y_argmin] else Y_argmin
        Y_mean = signal.mean()
        s = signal.std()
        if s > 0:
            G = np.abs(signal[Y_argopt] - Y_mean) / s
            G_crit = np.sqrt(self.crit_value ** 2 / (N - 2 + self.crit_value ** 2)) * float(N - 1) / np.sqrt(N)
            if G > G_crit:
                if replace:
                    signal[Y_argopt] = np.random.normal(loc = Y_mean, scale = s)
                return True
            else:
                return False
        else:
            return False

cdef class AnomalyDetector:
    cdef size_t window_size
    cdef cnp.float_t alpha
    cdef cnp.double_t crit_value

    def __cinit__(self, window_size, alpha = 0.01):
        self.window_size = window_size
        self.alpha = alpha
        self.crit_value = stats.t.isf(alpha / float(2 * window_size), window_size - 2)
    property window_size:
        def __get__(self): return self.window_size
    property alpha:
        def __get__(self): return self.alpha
    property crit_value:
        def __get__(self): return self.crit_value


cpdef cnp.ndarray getMissingValuesIndexes(cnp.ndarray data, double missing_value):
    return np.where(data == missing_value)[0]

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