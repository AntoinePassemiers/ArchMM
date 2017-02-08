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
from libc.string cimport memset
from libc.time cimport time
from cpython.buffer cimport PyObject_CheckBuffer


class ArrayTypeError(Exception): pass
class DataDimensionError(Exception): pass

cdef float cRand() nogil:
    return <float>rand() / <float>RAND_MAX

cdef int cRandint(Py_ssize_t start, Py_ssize_t end) nogil:
    cdef Py_ssize_t rang = end - start
    return <int>(cRand() * rang + start)

cdef void ensure_PyObject_Buffer(object data):
    if not PyObject_CheckBuffer(data):
        printf("Error : the sequence must implement the buffer interface\n")
        exit(EXIT_FAILURE)

def seed(value):
    srand(value)
    np.random.seed(seed = value)