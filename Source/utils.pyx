# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport memset
from cpython.buffer cimport PyObject_CheckBuffer
from libc.time cimport time

srand(time(NULL))

cdef float cRand():
	return <float>rand() / <float>RAND_MAX

cdef int cRandint(Py_ssize_t start, Py_ssize_t end):
	cdef Py_ssize_t rang = end - start
	return <int>(cRand() * rang + start)

cdef ensure_PyObject_Buffer(object data):
	if not PyObject_CheckBuffer(data):
		printf("Error : the sequence must implement the buffer interface\n")
		exit(EXIT_FAILURE)