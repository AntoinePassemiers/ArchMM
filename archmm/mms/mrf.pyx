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

cimport libc.math
from libc.stdlib cimport *
from libc.string cimport memset
from cython cimport view


"""
Extract features : Gabor feature, MRSAR feature, ...
"""

def softmax(vec):
    Z = np.exp(vec)
    return Z / Z.sum()

def gaussian_parameters_set(n_classes, n_variables):
    return {
        "n"     : np.empty(n_classes, dtype = np.int),
        "mu"    : np.empty((n_classes, n_variables), dtype = np.double),
        "inv_sigma" : np.empty((n_classes, n_variables, n_variables), dtype = np.double),
        "det"   : np.empty(n_classes, dtype = np.double)
    }

cdef inline size_t weighted_rng(cnp.float_t[:] weights) nogil:
    cdef float r = <float>rand() / <float>RAND_MAX
    cdef size_t i
    cdef float threshold = 0.0
    for i in range(weights.shape[0]):
        threshold += weights[i]
        if r < threshold:
            return i

clique_1st_order = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
clique_2nd_order = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]


cdef class MarkovRandomField:
    cdef size_t n_threads
    cdef size_t n_classes
    cdef bint is_warm
    cdef cnp.ndarray clique_structure
    cdef size_t n_channels
    cdef double threshold
    cdef cnp.float_t[:] label_weights
    cdef dict parameters

    def __init__(self, n_classes, clique = clique_1st_order, threshold = 0.0, n_threads = 1):
        """
        Attributes
        ----------
        n_classes: size_t
            Number of distinct labels in the training set
        is_warm: bool
            Indicates whether the parameters have been learned from
            a training set or not. The system must be warm before
            to be able to apply the simulated annealing algorithm
            on the test set.
        clique_structure: np.ndarray
            Inspired by scipy.ndimage.measurements.label
            Defines node connections inside a clique.
            examples: [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
                      -> First order clique in a grayscale (2D) image
                      [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
                      -> Second order clique in a grayscale (2D) image
        n_channels: size_t
            Number of channels in the input images
            Example: 1 for a grayscale image, 4 for a rgba image
        threshold: double
            Minimum energy change per iteration to prevent the system
            from freezing
        n_threads: size_t
            Number of threads to use during training and segmentation

        References
        ----------
        https://inf.u-szeged.hu/~ssip/2008/presentations2/Kato_ssip2008.pdf
        """
        self.n_threads = n_threads
        self.n_classes = n_classes
        self.is_warm = False
        self.clique_structure = np.asarray(clique, dtype = np.bool)
        self.threshold = threshold
        self.parameters = None

    property parameters:
        def __get__(self): return self.parameters
    
    def fit(self, X, Y):
        if type(X) != list:
            X, Y = [X], [Y]

        self.n_channels = 1 if len(X[0].shape) < 3 else X[0].shape[2]

        self.parameters = gaussian_parameters_set(self.n_classes, self.n_channels)
        for c in range(self.n_classes):
            pixels = [x[y == c] for x, y in zip(X, Y)]
            cluster = np.concatenate(pixels, axis = 0)
            
            self.parameters["n"][c] = len(cluster)
            self.parameters["mu"][c] = np.mean(cluster, axis = 0)
            sigma = np.cov(cluster.T)
            self.parameters["inv_sigma"][c] = np.linalg.inv(sigma)
            self.parameters["det"][c] = np.linalg.det(sigma)
            assert(not np.isnan(self.parameters["inv_sigma"][c]).any())

        self.label_weights = np.asarray(self.parameters["n"] / float(self.parameters["n"].sum()), dtype = np.float)
        self.is_warm = True

    cdef inline double singleton_potential(self, cnp.uint8_t[:] pixel, cnp.double_t[:] mu, cnp.double_t[:] d, 
                                           cnp.double_t[:, :] inv_sigma, double det) nogil:
        cdef double mahalanobis = 0.0
        cdef Py_ssize_t i, j
        for i in range(pixel.shape[0]):
            d[i] = pixel[i] - mu[i]
        for i in range(pixel.shape[0]):
            for j in range(pixel.shape[0]):
                mahalanobis += d[i] * inv_sigma[i, j] * d[j]
        return 0.5 * mahalanobis + libc.math.log(libc.math.sqrt(det * libc.math.M_2_PI ** pixel.shape[0]))

    cdef inline double doubleton_potential(self, size_t pixel_label, size_t neighbor_label, double beta) nogil:
        if pixel_label == neighbor_label:
            return -beta
        else:
            return beta

    def simulated_annealing(self, img, T0 = 10.0, dq = 0.95, beta = 1.5, max_n_iter = 1000):
        """
        Parameters
        ----------
        T0: double
            Initial temperature
        dq: double
            Temperature decrease at each iteration
            T(t+1) = dq.T(t)
        beta: double
            Doubleton potential (measure of segment homogeneity)
        max_n_iter: size_t
            Maximum number of iterations
        """
        cdef cnp.uint8_t[:, :, :] X
        if self.n_channels == 1:
            X = img.reshape(img.shape[0], img.shape[1], 1)
        else:
            X = img[:, :, :]
        cdef int[:, :] omega = np.random.randint(0, self.n_classes, size = img.shape[:2], dtype = np.int)
        cdef Py_ssize_t i, j
        cdef cnp.double_t[:, :] mus = self.parameters["mu"]
        cdef cnp.double_t[:, :, :] inv_sigmas = self.parameters["inv_sigma"]
        cdef cnp.double_t[:] dets = self.parameters["det"]
        cdef cnp.double_t[:] buf = np.empty(mus.shape[1], dtype = np.double)
        cdef size_t c # random label
        cdef cnp.double_t[:, :] potentials = np.full(img.shape[:2], np.finfo('d').max, dtype = np.double)
        cdef double potential, energy
        cdef double temperature = T0
        cdef double temperature_decrease_factor = dq
        cdef double cbeta = beta
        cdef size_t n_iter = max_n_iter
        cdef size_t iteration = 0
        with nogil:
            while iteration < n_iter:
                energy = 0.0
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        c = weighted_rng(self.label_weights)
                        potential = self.singleton_potential(X[i, j, :], mus[c, :], buf, inv_sigmas[c, :, :], dets[c])
                        if i > 0:
                            potential += self.doubleton_potential(c, omega[i-1, j], cbeta)
                        if i < X.shape[0] - 1:
                            potential += self.doubleton_potential(c, omega[i+1, j], cbeta)
                        if j > 0:
                            potential += self.doubleton_potential(c, omega[i, j-1], cbeta)
                        if j < X.shape[1] - 1:
                            potential += self.doubleton_potential(c, omega[i, j+1], cbeta)
                        if potential < potentials[i, j]:
                            potentials[i, j] = potential
                            omega[i, j] = c
                        # TODO : elif activation energy is sufficient
                temperature *= temperature_decrease_factor
                iteration += 1
                with gil:
                    print("Iteration %i" % iteration)
        return np.asarray(omega)

    def predict(self, X, method = "sa"):
        """
        Parameters
        ----------
        X: np.ndarray
            Image to process
        method: str
            Algorithm to apply on image X
            sa: simulated annealing (supervised)
            em: expectation-maximization (unsupervised)
        """
        method = method.lower()
        assert(method in ["sa", "em"])
        if method == "sa":
            assert(self.is_frozen)
        pass # TODO
    