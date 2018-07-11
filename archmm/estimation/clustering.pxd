# -*- coding: utf-8 -*-
# distutils: language=c

cimport numpy as cnp

cdef class ClusterSet:
    cdef cnp.double_t[:, :] data
    cdef cnp.int_t[:, :] clusters
    cdef size_t* cluster_sizes
    cdef size_t n_clusters, n_points, point_dim

    cdef void insert(self, int cluster_id, int point_id) nogil
    cdef void cluster_mean(self, cnp.double_t[:] buf, int cluster_id) nogil
    cdef size_t get_n_clusters(self) nogil
    cdef cnp.int16_t[:] get_labels(self)
    cdef unsigned int is_cluster_empty(self, int cluster_id) nogil

cdef ClusterSet perform_step(cnp.double_t[:, :] data,
                             cnp.double_t[:, :] centroids,
                             int n_clusters)
cdef cnp.ndarray init_centroids(cnp.ndarray data, int k, method=*)
cdef k_means_one_run(data, k, n_iter, init)
