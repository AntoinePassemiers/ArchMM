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
from functools import reduce

cimport libc.math
from libc.stdlib cimport *
from libc.string cimport memset
from cython cimport view


cdef class MarkovChainModel:
    cdef size_t n_symbols
    cdef size_t n_threads
    cdef double[:] initial_probs
    cdef double[:, :] transition_probs

    def __init__(self, n_symbols, n_threads = 1):
        self.n_symbols = n_symbols
        self.n_threads = n_threads

    cdef fit_sequence(self, int[:] sequence):
        cdef Py_ssize_t j
        with nogil:
            self.initial_probs[<int>sequence[0]] += 1
            for j in range(1, sequence.shape[0]):
                self.transition_probs[sequence[j-1], sequence[j]] += 1

    def fit(self, X):
        self.initial_probs = np.zeros(self.n_symbols, dtype = np.double)
        self.transition_probs = np.zeros((self.n_symbols, self.n_symbols), dtype = np.double)
        cdef size_t n_sequences = len(X)
        cdef Py_ssize_t i, n_transitions = 0
        with nogil:
            for i in prange(n_sequences, num_threads=self.n_threads):
                with gil:
                    self.fit_sequence(X[i])
                    n_transitions += len(X[i]) - 1

        self.initial_probs = np.asarray(self.initial_probs) / <double>len(X)
        self.transition_probs = np.asarray(self.transition_probs) / <double>n_transitions

    cdef double score_sequence(self, int[:] sequence):
        cdef double cost = 0.0
        cdef Py_ssize_t j
        with nogil:
            cost = libc.math.log2(self.initial_probs[sequence[0]])
            for j in range(1, sequence.shape[0]):
                cost += libc.math.log2(self.transition_probs[sequence[j-1], sequence[j]])
        return cost
    
    def generate(self, int n_samples):
        cdef int[:] sequence = np.empty(n_samples, dtype = np.int)
        cdef Py_ssize_t i
        sequence[0] = np.random.randint(0, self.n_symbols)
        for i in range(n_samples):
            pass # TODO

    def score(self, X):
        cdef double[:] costs = np.zeros(len(X), dtype = np.double)
        cdef size_t n_sequences = len(X)
        cdef Py_ssize_t i
        with nogil:
            for i in prange(n_sequences, num_threads = self.n_threads):
                with gil:
                    costs[i] = self.score_sequence(X[i])
        if len(X) == 1:
            return costs[0]
        else:
            return np.asarray(costs)


cdef class MarkovChainClassifier:
    cdef size_t n_symbols
    cdef size_t n_classes
    cdef list   models

    def __init__(self, n_classes, n_symbols):
        self.n_symbols = n_symbols
        self.n_classes = n_classes
        self.models = list()
        for i in range(n_classes):
            self.models.append(MarkovChainModel(n_symbols))
    
    def fit(self, X, y):
        for label in range(self.n_classes):
            sub_X = list()
            for k in range(len(y)):
                if y[k] == label:
                    sub_X.append(X[k])
            self.models[label].fit(sub_X)

    def score(self, X):
        scores = np.empty((len(X), self.n_classes), dtype=np.double)
        for i in range(self.n_classes):
            scores[:, i] = self.models[i].score(X)
        return np.asarray(scores)

    def classify(self, X):
        scores = self.score(X)
        return np.argmax(scores, axis = 1)
