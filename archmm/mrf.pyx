# -*- coding: utf-8 -*-
# mrf.pyx: Markov Random Fields for image segmentation
# author : Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport libc.math
from libc.stdlib cimport *
from libc.string cimport memset


FIRST_ORDER_CLIQUE = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
SECOND_ORDER_CLIQUE = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]


cdef double CNP_INF = <double>np.inf


cdef inline double singleton_potential(cnp.double_t[:] sample,
                                       cnp.double_t[:] mu, cnp.double_t[:] d, 
                                       cnp.double_t[:, :] inv_sigma, double det) nogil:
    cdef double mahalanobis = 0.0
    cdef Py_ssize_t i, j
    for i in range(sample.shape[0]):
        d[i] = sample[i] - mu[i]
    for i in range(sample.shape[0]):
        for j in range(sample.shape[0]):
            mahalanobis += d[i] * inv_sigma[i, j] * d[j]
    return 0.5 * mahalanobis + libc.math.log(libc.math.sqrt(det * libc.math.M_2_PI ** sample.shape[0]))


cdef inline double doubleton_potential(size_t pixel_label, size_t neighbor_label, double beta) nogil:
    return -beta if pixel_label == neighbor_label else beta


cdef inline double neighborhood_doubleton_potential(
    size_t i, size_t j, cnp.int_t[:, :] omega, cnp.int_t[:, :] clique, float beta, size_t c) nogil:
    cdef int hh = clique.shape[0] // 2 # Half clique height
    cdef int hw = clique.shape[1] // 2 # Half clique width
    cdef Py_ssize_t k, l, n_neighbors = 0
    cdef double potential = 0.0
    for k in range(<int>libc.math.fmax(<double>i-hh, <double>0), \
            <int>libc.math.fmin(<double>omega.shape[0], <double>i+hh+1)):
        for l in range(<int>libc.math.fmax(<double>j-hw, <double>0), \
                <int>libc.math.fmin(<double>omega.shape[1], <double>j+hw+1)):
            if k != i or l != j:
                potential += doubleton_potential(c, omega[k, l], beta)
                n_neighbors += 1
    if n_neighbors > 0:
        potential /= <double>n_neighbors
    return potential


def simulated_annealing(parameters, img, label_weights, clique=FIRST_ORDER_CLIQUE,
    float T0=10.0, float dq=0.95, float beta=50.0, size_t max_n_iter=10):
    """
    Args:
        T0 (double):
            Initial temperature
        dq (double):
            Temperature decrease at each iteration
            T(t+1) = dq.T(t)
        beta (double):
            Doubleton potential (measure of segment homogeneity)
        max_n_iter (size_t):
            Maximum number of iterations
    """
    cdef Py_ssize_t i, j, k, l, a, iteration
    cdef cnp.int_t c
    cdef cnp.double_t[:, :, :] X = img
    cdef float eta
    cdef double potential, dU, lowest_doubleton_potential, doubleton_potential
    cdef double[:, :] potentials = np.full(img.shape[:2], np.finfo('d').max, dtype=np.double)
    cdef cnp.double_t[:, :] mus = parameters["mu"]
    cdef cnp.double_t[:, :, :] inv_sigmas = parameters["inv_sigma"]
    cdef cnp.double_t[:] dets = parameters["det"]
    cdef cnp.double_t[:] buf = np.empty(mus.shape[1], dtype=np.double)
    cdef Py_ssize_t n_classes = len(mus)
    cdef cnp.int_t[:, :] omega = np.random.randint(0, n_classes, size=img.shape[:2], dtype=np.int)
    cdef cnp.int_t[:, :] previous_omega = np.copy(np.asarray(omega))
    cdef double temperature = T0
    temperature_decrease_factor = dq
    cdef cnp.int_t[:, :] random_numbers
    cdef cnp.float_t[:, :] random_etas
    cdef cnp.int_t[:, :] _clique = np.asarray(clique, dtype=np.int)

    cdef cnp.double_t[:, :, :] singleton_potentials = np.empty(
        (omega.shape[0], omega.shape[1], n_classes), dtype=np.double)
    cdef cnp.double_t[:, :, :] doubleton_potentials = np.empty(
        (omega.shape[0], omega.shape[1], n_classes), dtype=np.double)
    cdef cnp.double_t[:, :, :] total_potentials = np.empty(
        (omega.shape[0], omega.shape[1], n_classes), dtype=np.double)

    with nogil:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for c in range(n_classes):
                    singleton_potentials[i, j, c] = singleton_potential(
                        X[i, j, :], mus[c, :], buf, inv_sigmas[c, :, :], dets[c])

    for iteration in range(max_n_iter):
        # Generate a sufficient number of random numbers while GIL is still enabled
        random_numbers = np.asarray(
            np.random.choice(np.arange(len(label_weights)),
            size=(img.shape[0], img.shape[1]), p=label_weights), dtype=np.int)
        random_etas = np.asarray(np.random.rand(img.shape[0], img.shape[1]), dtype=np.float)
        with nogil:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    c = random_numbers[i, j]
                    doubleton_potential = neighborhood_doubleton_potential(
                        i, j, omega, _clique, beta, c)
                    doubleton_potentials[i, j, c] = doubleton_potential
                    potential = singleton_potentials[i, j, c] + doubleton_potentials[i, j, c]
                    
                    dU = potential - potentials[i, j]
                    eta = random_etas[i, j]
                    if (dU <= 0) or (eta < libc.math.exp(-dU / temperature)):
                        #if (dU <= 0) or (eta < 0.3):
                        potentials[i, j] = potential
                        total_potentials[i, j, c] = potential
                        omega[i, j] = c
        temperature *= temperature_decrease_factor
        print("Iteration %i" % iteration)
        if (np.asarray(previous_omega) == np.asarray(omega)).all():
            break # Thermodynamic system is frozen
        previous_omega[:, :] = omega
    return np.asarray(omega), np.asarray(total_potentials), \
        np.asarray(singleton_potentials), np.asarray(doubleton_potentials)


class MarkovRandomField:

    def __init__(self, clique=FIRST_ORDER_CLIQUE, 
        max_n_iter=50, beta=0.07, t0=10.0, dq=0.95, threshold=0.0, n_threads=1):
        """
        Args:
            threshold (double):
                Minimum energy change per iteration to prevent the system
                from freezing
            n_threads (size_t):
                Number of threads to use during training and segmentation
        
        Attributes:
            n_classes (size_t):
                Number of distinct labels in the training set
            is_warm (bool):
                Indicates whether the parameters have been learned from
                a training set or not. The system must be warm before
                to be able to apply the simulated annealing algorithm
                on the test set.
            n_channels (size_t):
                Number of channels in the input images
                Example: 1 for a grayscale image, 4 for a rgba image

        References:
            Zoltan Kato, Markov Random Fields in Image Segmentation
            https://inf.u-szeged.hu/~ssip/2008/presentations2/Kato_ssip2008.pdf
        """
        self.clique = clique

        self.n_classes = 2
        self.beta = beta
        self.t0 = t0
        self.dq = dq
        self.max_n_iter = max_n_iter
        self.threshold = threshold
        self.n_threads = n_threads
        self.is_warm = False

        self.n_channels = 0
        self.parameters = None
        self.label_weights = None

    def fit(self, X, Y):
        if len(X[0].shape) < 3:
            self.n_channels = 1
            X = [x[..., np.newaxis] for x in X]
        else:
            self.n_channels = X[0].shape[2]
        
        self.parameters = self.gaussian_parameters_set(self.n_classes, self.n_channels)
        for c in range(self.n_classes):
            pixels = [x[y == c] for x, y in zip(X, Y)]
            cluster = np.concatenate(pixels, axis=0)
            cluster = cluster[~np.any(np.isnan(cluster), axis=1)]

            # TODO: exception if empty cluster
                
            self.parameters["n"][c] = len(cluster)
            self.parameters["mu"][c] = np.mean(cluster, axis=0)
            sigma = np.cov(cluster.T)

            if self.n_channels > 1:
                self.parameters["inv_sigma"][c] = np.linalg.inv(sigma)
                self.parameters["det"][c] = np.linalg.det(sigma)
            else:
                sigma = sigma.reshape(1, 1)
                self.parameters["inv_sigma"][c] = 1. / sigma
                self.parameters["det"][c] = sigma[0, 0]

            mask = np.isnan(self.parameters["inv_sigma"][c])
            self.parameters["inv_sigma"][c][mask] = 0
        
        self.label_weights = np.asarray(self.parameters["n"] / float(
            self.parameters["n"].sum()), dtype=np.float)
        self.is_warm = True

    def predict(self, X, rettype='labels'):
        rettype = rettype.strip().lower()

        # TODO: check X
        if self.n_channels == 1:
            X = [x[..., np.newaxis] for x in X]

        predictions = list()
        for x in X:
            omega, potentials, sp, dp = simulated_annealing(
                self.parameters, x, self.label_weights, clique=self.clique,
                beta=self.beta, max_n_iter=self.max_n_iter, T0=self.t0, dq=self.dq)
            
            if rettype == 'proba':
                proba = np.empty_like(potentials)
                for c in range(2):
                    proba[:, :, 1-c] = np.exp(potentials[:, :, c]) / np.sum(np.exp(potentials), axis=2)
                predictions.append(np.nan_to_num(proba))
            elif rettype == 'energy':
                predictions.append(np.asarray(potentials))
            elif rettype == 'sp':
                predictions.append(np.asarray(sp))
            elif rettype == 'dp':
                predictions.append(np.asarray(dp))
            else:
                predictions.append(np.asarray(omega))
        return predictions
    
    def gaussian_parameters_set(self, n_classes, n_variables):
        return {
            "n"     : np.empty(n_classes, dtype=np.int),
            "mu"    : np.empty((n_classes, n_variables), dtype=np.double),
            "inv_sigma" : np.empty((n_classes, n_variables, n_variables), dtype=np.double),
            "det"   : np.empty(n_classes, dtype=np.double)
        }