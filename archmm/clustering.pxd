# -*- coding: utf-8 -*-
# distutils: language=c

cimport numpy as cnp


cdef perform_step(cnp.ndarray data, cnp.ndarray centroids, size_t n_clusters)
cdef cnp.ndarray randomize_centroids(cnp.double_t[:, :] data, Py_ssize_t k)
