# -*- coding: utf-8 -*-
# iohmm.pxd: Input-Output Hidden Markov Model
# author: Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport libc.math

from archmm.hmm cimport *


cdef data_t MINUS_INF = np.nan_to_num(-np.inf)


cdef inline data_t elnproduct(data_t eln_x, data_t eln_y) nogil
cdef inline data_t elnsum(data_t eln_x, data_t eln_y) nogil


cdef class IOHMM(HMM):

    cdef bint is_classifier
    cdef int output_dim

    cdef data_t[:, :, :] A_c
    cdef cnp.int_t[:] T_s

    cdef object start_subnetwork
    cdef list transition_subnetworks
    cdef list emission_subnetworks

    cdef data_t[:, :] compute_ln_phi(self, int sequence_id, int t) nogil