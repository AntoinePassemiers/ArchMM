# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
cnp.import_array()

import multiprocessing
from cython.parallel import parallel, prange, threadid

from libc.stdlib cimport *
from libc.string cimport memset
from libc.math cimport sqrt
from cpython.buffer cimport PyBuffer_IsContiguous
from cython cimport view

from archmm.math cimport sample_gaussian, euclidean_distance


cdef cnp.double_t NP_INF_VALUE = <cnp.double_t>np.inf


cdef class ClusterSet:

    def __cinit__(self, data, n_clusters, n_points, point_dim):
        self.data = data
        self.n_clusters = n_clusters
        self.n_points = n_points
        self.point_dim = point_dim
        self.clusters = np.empty((n_clusters, n_points), dtype=int)
        self.cluster_sizes = <size_t*>calloc(n_clusters, sizeof(size_t))
            
    def __dealloc__(self):
        free(self.cluster_sizes)
        
    cdef void insert(self, int cluster_id, int point_id) nogil:
        """ Inserts [[element]] in the cluster [[cluster_id]].
        The cluster is implemented using arrays. """
        if cluster_id < self.n_clusters:
            self.clusters[cluster_id][self.cluster_sizes[cluster_id]] = point_id
            self.cluster_sizes[cluster_id] += 1
        else:
            with gil:
                raise MemoryError("Cluster max size exceeded")
        
    cdef void cluster_mean(self, cnp.double_t[:] buf, int cluster_id) nogil:
        """ Computes the mean of the cluster [[cluster_id]] by averaging the 
        contained points.
        """
        cdef int i, j, n_points = self.cluster_sizes[cluster_id]
        for j in range(self.point_dim):
            buf[j] = 0.0
            for i in range(n_points):
                buf[j] += self.data[self.clusters[cluster_id][i]][j]
        for j in range(self.point_dim):
            buf[j] /= n_points
        
    cdef size_t get_n_clusters(self) nogil:
        return self.n_clusters

    cdef cnp.int16_t[:] get_labels(self):
        cdef cnp.int16_t[:] labels = np.empty(self.data.shape[0], dtype=np.int16)
        with nogil:
            for i in range(self.n_clusters):
                for j in range(self.cluster_sizes[i]):
                    labels[self.clusters[i, j]] = i
        return labels
    
    cdef unsigned int is_cluster_empty(self, int cluster_id) nogil:
        return 1 if self.cluster_sizes[cluster_id] == 0 else 0
            
     
cdef ClusterSet perform_step(cnp.double_t[:, :] data,
                             cnp.double_t[:, :] centroids,
                             int n_clusters):
    """ Computes the euclidean distances between each point of the dataset and 
    each of the centroids, finds the closest centroid from the distances, and
    updates the centroids by averaging the new clusters
    
    Args:
        data : input dataset
        centroids : positions of the centroids
        clusters : objects containing all the information about each cluster
                   the points it contains, its centroid, etc.
    """
    cdef int n_dim = data.shape[1]
    cdef ClusterSet clusters = ClusterSet(data, n_clusters, len(data), n_dim)
    cdef int mu_index, i, k = 0
    cdef cnp.double_t min_distance, current_distance
    with nogil:
        for k in range(data.shape[0]):
            min_distance = NP_INF_VALUE
            for i in range(centroids.shape[0]):
                current_distance = euclidean_distance(data[k, :], centroids[i, :])
                if current_distance < min_distance:
                    min_distance = current_distance
                    mu_index = i
            clusters.insert(mu_index, k)
    for i in range(clusters.get_n_clusters()):
        if clusters.is_cluster_empty(i):
            clusters.insert(i, np.random.randint(0, len(data), size=1)[0])
    return clusters


cdef cnp.ndarray init_centroids(cnp.ndarray data, int k, method='gaussian'):
    """ Initializes the centroids by picking random points from the dataset
    
    Args:
        data : input dataset
        k : number of clusters (= number of centroids)
    """
    method = method.strip().lower()
    if method == 'uniform':
        indices = np.random.randint(0, len(data), size=k)
        centroids =  data[indices, :]
    else:
        mu = np.mean(data, axis=0)
        sigma = np.cov(data.T)
        inv_sigma = np.linalg.inv(sigma)
        centroids = sample_gaussian(mu, inv_sigma, k)
    return centroids


cdef k_means_one_run(data, k, n_iter, init):
    """ Implementation of the k-means algorithm """
    cdef int n_dim = data.shape[1]
    cdef int i, j
    cdef cnp.ndarray centroids = init_centroids(data, k, method=init)
    cdef cnp.ndarray old_centroids = np.zeros((k, n_dim), dtype=centroids.dtype)
    cdef size_t iterations = 0
    cdef ClusterSet clusters
    while not (iterations > n_iter or (old_centroids == centroids).all()):
        iterations += 1
        clusters = perform_step(
            data.astype(np.double), centroids.astype(np.double), k)
        for i in range(clusters.get_n_clusters()):
            old_centroids[i] = centroids[i]
            clusters.cluster_mean(centroids[i, :], i)
    labels = np.asarray(clusters.get_labels()) if iterations > 0 else None
    return centroids, np.asarray(labels)


def k_means(data, k, n_iter=10, init='gaussian', n_runs=5):
    best_cost = np.inf
    for i in range(n_runs):
        centroids, labels = k_means_one_run(data, k, n_iter=n_iter, init=init)
        # Evaluate k-means cost function
        cost = sum([np.sqrt(np.sum((data[labels==c]-centroids[c])**2,
            axis=1)).sum() for c in range(k)])
        if cost < best_cost:
            best_cost = cost
            result = (centroids, labels)
    return result
