# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport atoi, malloc, calloc, free

import multiprocessing
from cython.parallel import parallel, prange, threadid


NP_INF_VALUE = np.nan_to_num(np.inf)
cdef size_t INITIAL_CLUSTER_SIZE = 4

cdef class ClusterSet:
    
    cdef double*** clusters
    cdef size_t* cluster_sizes, 
    cdef size_t* cluster_max_sizes
    cdef size_t n_clusters, n_points, point_dim
    
    def __cinit__(self, n_clusters, n_points, point_dim):
        self.n_clusters = n_clusters
        self.n_points = n_points
        self.point_dim = point_dim
        self.clusters = <double***>malloc(n_clusters * sizeof(double**))
        cdef size_t k
        for k in range(n_clusters):
            self.clusters[k] = <double**>malloc(INITIAL_CLUSTER_SIZE * sizeof(double*))
        self.cluster_sizes = <size_t*>malloc(n_clusters * sizeof(size_t))
        self.cluster_max_sizes = <size_t*>malloc(n_clusters * sizeof(size_t))
        for k in range(n_clusters):
            self.cluster_sizes[k] = 0
            self.cluster_max_sizes[k] = INITIAL_CLUSTER_SIZE
            
    def __dealloc__(self):
        free(self.clusters)
        free(self.cluster_sizes)
        free(self.cluster_max_sizes)
        
    cdef void insert(self, Py_ssize_t cluster_id, cnp.double_t[:] element) nogil:
        """ Inserts [[element]] in the cluster [[cluster_id]].
        The cluster is implemented using an array. If the array is full,
        a new array is initialized with the double of the previous array's size. """
        cdef double** new_cluster
        cdef size_t i
        if cluster_id < self.n_clusters:
            if self.cluster_sizes[cluster_id] == self.cluster_max_sizes[cluster_id]:
                self.cluster_max_sizes[cluster_id] *= 2
                if self.cluster_max_sizes[cluster_id] > self.n_points:
                    self.cluster_max_sizes[cluster_id] = self.n_points
                new_cluster = <double**>malloc(self.cluster_max_sizes[cluster_id] * sizeof(double*))
                for i in range(self.cluster_sizes[cluster_id]):
                    new_cluster[i] = self.clusters[cluster_id][i]
                free(self.clusters[cluster_id])
                self.clusters[cluster_id] = new_cluster
            self.clusters[cluster_id][self.cluster_sizes[cluster_id]] = <double*>malloc(self.point_dim * sizeof(double))
            for i in range(self.point_dim):
                self.clusters[cluster_id][self.cluster_sizes[cluster_id]][i] = element[i]
            self.cluster_sizes[cluster_id] += 1
        
    cdef cnp.double_t[:] clusterMean(self, size_t cluster_id) nogil:
        """ Computes the mean of the cluster [[cluster_id]] by averaging the 
        contained points """ 
        # cdef cnp.double_t[:] cmean = np.zeros(self.point_dim, dtype = np.double)
        cdef cnp.double_t[:] cmean
        with gil:
            cmean = <cnp.double_t[:self.point_dim]>calloc(self.point_dim, sizeof(cnp.double_t))
        cdef size_t i, j
        cdef size_t n_points = self.cluster_sizes[cluster_id]
        for i in range(n_points):
            for j in range(self.point_dim):
                cmean[j] += self.clusters[cluster_id][i][j]
        for j in range(self.point_dim):
            cmean[j] /= n_points
        return cmean
        
    cdef size_t getNClusters(self) nogil:
        return self.n_clusters
    
    cdef unsigned int isClusterEmpty(self, Py_ssize_t cluster_id) nogil:
        return 1 if self.cluster_sizes[cluster_id] == 0 else 0
            
     
cdef perform_step(cnp.ndarray data, cnp.ndarray centroids, ClusterSet clusters):
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
    # cdef cnp.double_t[:, ::1] databuffer = data[:, :]
    cdef Py_ssize_t mu_index, i, k = 0
    cdef double min_distance = NP_INF_VALUE
    cdef double current_distance
    for k in range(len(data)):
        for i in range(len(centroids)):
            current_distance = np.linalg.norm(data[k] - centroids[i])
            if current_distance < min_distance:
                min_distance = current_distance
                mu_index = i
        clusters.insert(mu_index, data[k])
        k += 1
    for i in range(clusters.getNClusters()):
        if clusters.isClusterEmpty(i):
            clusters.insert(i, data[np.random.randint(0, len(data), size = 1)].flatten())
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
    for cluster in range(0, k):
        centroids[cluster, :] = data[int(np.random.randint(0, len(data), size = 1)), :]
    return centroids

cpdef kMeans(data, k, n_iter = 1000):
    """ Implementation of the well-known k-means algorithm """
    cdef Py_ssize_t n_dim = data.shape[1]
    cdef Py_ssize_t i
    cdef cnp.double_t[:, ::1] centroids = randomize_centroids(data, k)
    cdef cnp.ndarray old_centroids = np.empty((k, n_dim), dtype = np.double)
    cdef size_t iterations = 0
    cdef ClusterSet clusters
    while not (iterations > n_iter or old_centroids.all() == centroids.all()):
        iterations += 1
        clusters = ClusterSet(k, len(data), n_dim)
        clusters = perform_step(data, centroids, clusters)
        for i in range(clusters.getNClusters()):
            old_centroids[i] = centroids[i]
            centroids[i] = clusters.clusterMean(i)
    return centroids