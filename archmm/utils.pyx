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

import sys


class ArrayTypeError(Exception): pass
class DataDimensionError(Exception): pass
class DependencyError(Exception): pass
class NotImplementedError(Exception): pass
class NotImplementedAbstractMethodError(Exception): pass

""" Decorators """

def abstractmethod(func):
    def func_wrapper(*args):
        raise NotImplementedError(
            "%s abstract method must be implemented" % func.__name__
        )
    func_wrapper.__name__ = func.__name__
    return func_wrapper

def todo(func):
    def func_wrapper(*args):
        raise NotImplementedError("%s is not implemented yet" % func.__name__)
    func_wrapper.__name__ = func.__name__
    return func_wrapper


""" Cython utils """

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