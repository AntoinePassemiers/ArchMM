# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
import multiprocessing

cimport numpy as cnp
from libc.stdlib cimport *
from libc.string cimport memset
from libc.math cimport sqrt
from cython cimport view
from cpython.buffer cimport PyBuffer_IsContiguous

from cython.parallel import parallel, prange, threadid

include "Fuzzy.pyx"


NP_INF_VALUE = np.nan_to_num(np.inf)
cdef size_t INITIAL_CLUSTER_SIZE = 4

cdef class ClusterSet:
    
    cdef cnp.double_t[:, :] data
    cdef Py_ssize_t[:, :] clusters
    cdef size_t* cluster_sizes
    cdef size_t n_clusters, n_points, point_dim
    
    def __cinit__(self, data, n_clusters, n_points, point_dim):
        self.data = data
        self.n_clusters = n_clusters
        self.n_points = n_points
        self.point_dim = point_dim
        self.clusters = np.empty((n_clusters, n_points), dtype = int)
        self.cluster_sizes = <size_t*>calloc(n_clusters, sizeof(size_t))
            
    def __dealloc__(self):
        free(self.cluster_sizes)
        
    cdef void insert(self, Py_ssize_t cluster_id, Py_ssize_t point_id) nogil:
        """ Inserts [[element]] in the cluster [[cluster_id]].
        The cluster is implemented using arrays. """
        if cluster_id < self.n_clusters:
            self.clusters[cluster_id][self.cluster_sizes[cluster_id]] = point_id
            self.cluster_sizes[cluster_id] += 1
        else:
            with gil:
                raise MemoryError("Cluster max size exceeded")
        
    cdef cnp.double_t[:] clusterMean(self, size_t cluster_id) nogil:
        """ Computes the mean of the cluster [[cluster_id]] by averaging the 
        contained points """
        cdef cnp.double_t[:] cmean
        with gil:
            cmean = <cnp.double_t[:self.point_dim]>calloc(self.point_dim, sizeof(cnp.double_t))
        cdef size_t i, j
        cdef size_t n_points = self.cluster_sizes[cluster_id]
        for i in range(n_points):
            for j in range(self.point_dim):
                cmean[j] += self.data[self.clusters[cluster_id][i]][j]
        for j in range(self.point_dim):
            cmean[j] /= n_points
        return cmean
        
    cdef size_t getNClusters(self) nogil:
        return self.n_clusters
    
    cdef unsigned int isClusterEmpty(self, Py_ssize_t cluster_id) nogil:
        return 1 if self.cluster_sizes[cluster_id] == 0 else 0
            
     
cdef perform_step(cnp.ndarray data, cnp.ndarray centroids, size_t n_clusters):
    """ Computes the euclidean distances between each point of the dataset and 
    each of the centroids, finds the closest centroid from the distances, and
    updates the centroids by averaging the new clusters
    
    Parameters
    ----------
    data : input dataset
    centroids : positions of the centroids
    clusters : objects containing all the information about each cluster
               the points it contains, its centroid, etc.
    """
    cdef Py_ssize_t n_dim = data.shape[1]
    cdef ClusterSet clusters = ClusterSet(data, n_clusters, len(data), n_dim)
    cdef Py_ssize_t mu_index, i, k = 0
    cdef double min_distance = NP_INF_VALUE
    cdef double current_distance
    for k in range(len(data)):
        for i in range(len(centroids)):
            current_distance = np.linalg.norm(data[k] - centroids[i])
            if current_distance < min_distance:
                min_distance = current_distance
                mu_index = i
        clusters.insert(mu_index, k)
        k += 1
    for i in range(clusters.getNClusters()):
        if clusters.isClusterEmpty(i):
            clusters.insert(i, np.random.randint(0, len(data), size = 1)[0])
    return clusters

cdef cnp.ndarray randomize_centroids(cnp.double_t[:, :] data, Py_ssize_t k):
    """ Initializes the centroids by picking random points from the dataset
    
    Parameters
    ----------
    data : input dataset
    k : number of clusters (= number of centroids)
    """
    cdef Py_ssize_t n_dim = data.shape[1]
    cdef cnp.ndarray centroids = np.empty((k, n_dim), dtype = np.double)
    cdef Py_ssize_t random_index
    for cluster in range(0, k):
        random_index = int(np.random.randint(0, len(data), size = 1))
        centroids[cluster, :] = data[random_index, :]
    return centroids

cpdef kMeans(data, k, n_iter = 100):
    """ Implementation of the k-means algorithm """
    cdef Py_ssize_t n_dim = data.shape[1]
    cdef Py_ssize_t i
    cdef cnp.ndarray centroids = randomize_centroids(data, k)
    cdef cnp.ndarray old_centroids = np.empty((k, n_dim), dtype = np.double)
    cdef size_t iterations = 0
    cdef ClusterSet clusters
    while not (iterations > n_iter or old_centroids.all() == centroids.all()):
        iterations += 1
        clusters = perform_step(data, centroids, k)
        for i in range(clusters.getNClusters()):
            old_centroids[i] = centroids[i]
            centroids[i] = clusters.clusterMean(i)
        del clusters
    return centroids

cpdef fuzzyCMeans(data, n_clusters, n_iter = 100, fuzzy_coefficient = 2.0):
    """ Implementation of the fuzzy C-means algorithm """
    cdef double m = fuzzy_coefficient
    cdef size_t iterations = 0
    cdef size_t max_iterations = n_iter
    cdef Py_ssize_t n_dim = data.shape[1]
    cdef size_t N = data.shape[0]
    cdef size_t C = n_clusters
    cdef cnp.double_t[:, :] X = data
    cdef cnp.ndarray g = np.empty((C, n_dim), dtype = np.double)
    cdef cnp.double_t[::view.strided, ::1] centroids = np.ascontiguousarray(g)
    cdef dom_t[::view.strided, ::1] U = np.random.rand(N, C)
    U /= U.sum(axis = 1)[:, np.newaxis]
    cdef double denom, denomnum, denomdenom
    cdef Py_ssize_t i, j, k, l
    with nogil:
        while iterations < max_iterations: # TODO
            iterations += 1
            memset(<void*>&U[0, 0], 0x00, N * C * sizeof(dom_t))
            for j in range(C):
                for i in range(N):
                    for l in range(n_dim):
                        centroids[j, l] += U[i, j] * X[i, l]
                denom = 0.0
                for i in range(N):
                    denom += U[i, j]
                for l in range(n_dim):
                    centroids[j, l] /= denom
            for j in range(C):
                for i in range(N):
                    denomnum = 0.0
                    for l in range(n_dim):
                        denomnum += (X[i, l] - centroids[j, l]) ** 2
                    denomnum = sqrt(denomnum)
                    denom = 0.0
                    for k in range(C):
                        denomdenom = 0.0
                        for l in range(n_dim):
                            denomdenom += (X[i, l] - centroids[k, l]) ** 2
                        denomdenom = sqrt(denomdenom)
                        denom += (denomnum / denomdenom) ** -(2.0 / m - 1.0)
    return np.asarray(centroids), np.asarray(U)