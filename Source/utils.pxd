# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport memset
from cpython.buffer cimport PyObject_CheckBuffer
from libc.time cimport time


cdef float cRand()
cdef int cRandint(Py_ssize_t start, Py_ssize_t end)
cdef ensure_PyObject_Buffer(object data)