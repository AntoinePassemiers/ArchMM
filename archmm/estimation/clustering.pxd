# -*- coding: utf-8 -*-
# distutils: language=c

cimport numpy as cnp

cdef class ClusterSet:
    cdef cnp.double_t[:, :] data
    cdef int[:, :] clusters
    cdef size_t* cluster_sizes
    cdef size_t n_clusters, n_points, point_dim

    cdef void insert(self, Py_ssize_t cluster_id, Py_ssize_t point_id) nogil
    cdef cnp.double_t[:] clusterMean(self, size_t cluster_id) nogil
    cdef size_t getNClusters(self) nogil
    cdef cnp.int16_t[:] getLabels(self)
    cdef unsigned int isClusterEmpty(self, Py_ssize_t cluster_id) nogil

cdef ClusterSet perform_step(cnp.ndarray data, cnp.ndarray centroids, size_t n_clusters)
cdef cnp.ndarray randomize_centroids(cnp.double_t[:, :] data, Py_ssize_t k)
