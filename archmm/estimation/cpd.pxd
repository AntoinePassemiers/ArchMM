# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "kernel_.h":
    ctypedef double rbf_distance_t
    ctypedef double data_distance_t

    inline rbf_distance_t fast_gaussianRBF(data_distance_t r, float epsilon) nogil
    inline rbf_distance_t fast_multiquadricRBF(data_distance_t r, float epsilon) nogil
    inline rbf_distance_t fast_inverseQuadraticRBF(data_distance_t r, float epsilon) nogil
    inline rbf_distance_t fast_inverseMultiquadricRBF(data_distance_t r, float epsilon) nogil
    inline rbf_distance_t fast_polyharmonicSplineRBF(data_distance_t r, double k) nogil
    inline rbf_distance_t fast_thinPlateSplineRBF(data_distance_t r, double k) nogil