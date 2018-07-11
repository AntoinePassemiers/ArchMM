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

    cdef cnp.int_t[:, ::1] transition_mask
    cdef data_t[::1] initial_probs
    cdef data_t[:, ::1] transition_probs
    cdef data_t[::1] ln_initial_probs
    cdef data_t[:, ::1] ln_transition_probs

    cdef data_t[::1] sample_one_from_state(self, int state_id) nogil
    cdef data_t forward(self,
                        data_t[:, ::1] lnf,
                        data_t[:, ::1] ln_alpha,
                        data_t[::1] tmp) nogil
    cdef data_t backward(self,
                         data_t[:, ::1] lnf,
                         data_t[:, ::1] ln_beta,
                         data_t[::1] tmp)
    cdef e_step(self,
                data_t[:, ::1] lnf,
                data_t[:, ::1] ln_alpha,
                data_t[:, ::1] ln_beta,
                data_t[:, ::1] ln_gamma,
                data_t[:, :, ::1] ln_xi)