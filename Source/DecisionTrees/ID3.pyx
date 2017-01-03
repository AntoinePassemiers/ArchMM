# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport *
cimport libc.math

ctypedef double forwardBackwardProbs_t

cdef inline double piStateCost(forwardBackwardProbs_t* gamma0, float* p, size_t n_states):
    """
    Computes the cost of the initial hidden state

    Attributes
    ---------
    float* p : probabilities of starting from each one of the hidden states
    size_t num_classes : number of different labels in the dataset
    """
    cdef Py_ssize_t i
    cdef double cost = 0.0
    with nogil:
        for i in range(n_states):
            cost -= gamma0[i] * libc.math.log(p[i])
    return cost

cdef inline double stateTransitionCost(forwardBackwardProbs_t* xi, float* p, size_t n_states):
    """
    Computes the cost of the current hidden state

    Attributes
    ---------
    float* p : probabilities of moving from the previous hidden state to the current one
    size_t num_classes : number of different labels in the dataset
    """
    cdef Py_ssize_t i
    cdef double cost = 0.0
    with nogil:
        for i in range(n_states):
            cost -= xi[i] * libc.math.log(p[i])
    return cost
