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


class Version:
    def __init__(self, major = None, minor = None, build = None, revision = None):
        if major == VERSION_ANY:
            self.major = self.minor = self.build = self.revision = VERSION_ANY
        else:
            self.major = major
            self.minor = minor
            self.build = build
            self.revision = revision
    def __str__(self):
        if self.major == VERSION_ANY:
            return "v.v.v"
        v = ".".join([self.major, self.minor])
        if self.build:
            v = ".".join([v, self.build])
            if self.revision:
                v = ".".join([v, self.revision])
        return v

VERSION_ANY = np.inf

try:
    import theano
    import theano.tensor
    USE_THEANO = True # theano.__version__
    THEANO_VERSION = Version(VERSION_ANY)
except ImportError:
    USE_THEANO = False
    THEANO_VERSION = Version(None)
try:
    import cvxpy
    USE_CVXPY = True
    CVXPY_VERSION = Version(VERSION_ANY)
except ImportError:
    USE_CVXPY = False
    CVXPY_VERSION = Version(None)
try:
    try:
        import matplotlib.pyplot as plt
    except AttributeError:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    USE_PYPLOT = True
    PYPLOT_VERSION = Version(VERSION_ANY)
except ImportError:
    USE_PYPLOT = False
    PYPLOT_VERSION = Version(None)

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

def genericRequirementFunction(package_name, test_variable, test_version):
    def requiresPackage(version):
        def requiresPackage_decorator(func):
            a = isinstance(version, Version) # TODO
            if test_variable and test_version > version:
                return func
            else:
                def func_wrapper(*args, **kwargs):
                    s = (package_name, func.__name__)
                    raise DependencyError("Error : requires %s to call %s." % (s, func.__name__))
                return func_wrapper
        return requiresPackage_decorator
    return requiresPackage

requiresTheano = genericRequirementFunction("Theano", USE_THEANO, THEANO_VERSION)
requiresCvxPy  = genericRequirementFunction("cvxpy", USE_CVXPY, CVXPY_VERSION)
requiresPyplot = genericRequirementFunction("matplotlib", USE_PYPLOT, PYPLOT_VERSION)


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