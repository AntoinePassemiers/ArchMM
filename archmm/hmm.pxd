# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()


ctypedef cnp.double_t data_t


cdef class HMM:

    cdef str arch
    cdef int n_states
    cdef int n_features

    cdef cnp.int_t[:, :] transition_mask
    cdef data_t[:] initial_probs
    cdef data_t[:, :] transition_probs
    cdef data_t[:] ln_initial_probs
    cdef data_t[:, :] ln_transition_probs

    cdef data_t[:, :] compute_ln_phi(self, int sequence_id, int t) nogil
    cdef data_t[:] sample_one_from_state(self, int state_id) nogil
    cdef data_t forward(self,
                        data_t[:, :] lnf,
                        data_t[:, :] ln_alpha,
                        data_t[:] tmp,
                        int sequence_id) nogil
    cdef data_t backward(self,
                         data_t[:, :] lnf,
                         data_t[:, :] ln_beta,
                         data_t[:] tmp,
                         int sequence_id)
    cdef e_step(self,
                data_t[:, :] lnf,
                data_t[:, :] ln_alpha,
                data_t[:, :] ln_beta,
                data_t[:, :] ln_gamma,
                data_t[:, :, :] ln_xi,
                int sequence_id)


cdef class GHMM(HMM):

    cdef data_t[:, :] mu
    cdef data_t[:, :, :] sigma

    cdef data_t[:] sample_one_from_state(self, int state_id) nogil


cdef class GMMHMM(HMM):

    cdef int n_components
    cdef data_t[:, :] weights
    cdef data_t[:, :, :] mu
    cdef data_t[:, :, :, :] sigma

    cdef data_t[:] sample_one_from_state(self, int state_id) nogil


cdef class MHMM(HMM):

    cdef int n_unique
    cdef data_t[:, :] proba

    cdef data_t[:] sample_one_from_state(self, int state_id) nogil
