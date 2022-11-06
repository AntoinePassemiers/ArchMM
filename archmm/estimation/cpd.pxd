# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp
cnp.import_array()


cdef double NUMPY_INF_VALUE = np.nan_to_num(np.inf)

cdef unsigned int POLYNOMIAL_APRX = 1
cdef unsigned int WAVELET_APRX = 2
cdef unsigned int FOURIER_APRX = 3

cdef unsigned int SUM_OF_SQUARES_COST = 4
cdef unsigned int MAHALANOBIS_DISTANCE_COST = 5

cdef unsigned int KERNEL_RADIAL = 100
cdef unsigned int KERNEL_TRICUBIC = 101


cdef extern from "estimation/queue_.h":
    struct Iteration:
        int begin
        int end
        double previous_cost
    struct QueueNode:
        void* data
        QueueNode* next
    struct Queue:
        QueueNode* front_node
        QueueNode* rear_node
        size_t length
    Queue* newQueue() nogil
    void enqueue(Queue* queue, void* data) nogil
    void* dequeue(Queue* queue) nogil
    int isQueueEmpty(Queue* queue) nogil


cdef extern from "estimation/kernel_.h":
    ctypedef double rbf_distance_t
    ctypedef double data_distance_t

    rbf_distance_t fast_gaussianRBF(data_distance_t r, float epsilon) nogil
    rbf_distance_t fast_multiquadricRBF(data_distance_t r, float epsilon) nogil
    rbf_distance_t fast_inverseQuadraticRBF(data_distance_t r, float epsilon) nogil
    rbf_distance_t fast_inverseMultiquadricRBF(data_distance_t r, float epsilon) nogil
    rbf_distance_t fast_polyharmonicSplineRBF(data_distance_t r, double k) nogil
    rbf_distance_t fast_thinPlateSplineRBF(data_distance_t r, double k) nogil