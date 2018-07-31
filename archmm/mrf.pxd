# -*- coding: utf-8 -*-
# mrf.pxd: Markov Random Fields for image segmentation
# author : Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()


cdef inline double singleton_potential(cnp.double_t[:] sample,
                                       cnp.double_t[:] mu, cnp.double_t[:] d, 
                                       cnp.double_t[:, :] inv_sigma, double det) nogil

cdef inline double doubleton_potential(size_t pixel_label, size_t neighbor_label, double beta) nogil

cdef inline double neighborhood_doubleton_potential(
    size_t i, size_t j, cnp.int_t[:, :] omega, cnp.int_t[:, :] clique, float beta, size_t c) nogil