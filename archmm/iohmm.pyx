# -*- coding: utf-8 -*-
# iohmm.pyx : Input-Output Hidden Markov Model
# author: Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
cimport numpy as cnp
cnp.import_array()

import cython
from cython.parallel import parallel, prange
from threading import Thread

cimport libc.math
from libc.stdio cimport *

from archmm.ann.layers import *
from archmm.ann.subnetworks import *

from archmm.anomaly import *
from archmm.anomaly cimport *
from archmm.hmm cimport HMM, data_t
from archmm.hmm import np_data_t


ctypedef cnp.double_t prob_t
ctypedef cnp.double_t ln_prob_t

cdef ln_prob_t MINUS_INF = np.nan_to_num(-np.inf)

cdef inline prob_t eexp(ln_prob_t x) nogil:
    if x == MINUS_INF:
        return 0.0
    else:
        return libc.math.exp(x)

cdef inline ln_prob_t eln(prob_t x) nogil:
    if x <= 0:
        return MINUS_INF
    else:
        return libc.math.log(x)

cdef inline ln_prob_t elnproduct(ln_prob_t eln_x, ln_prob_t eln_y) nogil:
    if (eln_x == MINUS_INF) or (eln_y == MINUS_INF):
        return MINUS_INF
    else:
        return eln_x + eln_y

cdef inline ln_prob_t elnsum(ln_prob_t eln_x, ln_prob_t eln_y) nogil:
    if (eln_x == MINUS_INF) or (eln_y == MINUS_INF):
        if eln_x == MINUS_INF:
            return eln_y
        else:
            return eln_x
    else:
        return libc.math.log(libc.math.exp(eln_x) + libc.math.exp(eln_y))

def pyLogsum(x):
    return np.log(np.sum(np.exp(x)))



def create_buffer_list(X, shape, dtype):
    buffer_list = list()
    for sequence in X:
        buffer_shape = tuple([len(sequence)] + list(shape))
        buffer_list.append(np.empty(buffer_shape, dtype=dtype))
    return buffer_list


cdef class IOHMM(HMM):

    cdef bint is_classifier
    cdef int output_dim

    cdef data_t[:, :, :] A_c
    cdef cnp.int_t[:] T_s

    cdef object start_subnetwork
    cdef list transition_subnetworks
    cdef list emission_subnetworks

    def __init__(self, n_states, arch='ergodic'):
        HMM.__init__(self, n_states, arch=arch)
        self.A_c = None
        self.T_s = None
    
    cdef data_t[:, :] compute_ln_phi(self, int sequence_id, int t) nogil:
        return self.A_c[self.T_s[sequence_id]+t]

    def fit(self, X_s, y_s, max_n_iter=100):
        """
        Generalized Expectation-Maximization algorithm for training Input-Output Hidden Markov Models (IOHMM)
        The expectation part is implemented the old-fashioned way, like in a regular expectation-maximization
        algorithm. The maximization part is based on the MLP stochastic gradient descent.

        Args:
            inputs (list):
                list of input sequences, where each sequence is a 3D buffer
                Shape of each sequence : (n_samples, n_features),
                where n_samples is the length of the sequence
                ans n_features is the dimensionality of the input
                n_samples can vary from one sequence to another
            targets (np.ndarray):
                array of labels for classification (length = number of sequences)
            n_states (int):
                number of hidden states
            is_classifier (bool):
                if true, the model will be a classifier
                if false, the model will be a regressor
            n_classes (int):
                number of distinct labels to consider for classification tasks
            parameters (IOConfig):
                parameters of the model
                see Core.py for details
        """
        cdef int i, j, t, k, l, p, iteration, seq_length, start
        n_sequences = len(X_s)
        assert(n_sequences == len(y_s))

        n_features = X_s[0].shape[1]
        n_classes = len(np.unique(np.concatenate(y_s, axis=0)))

        loglikelihood = np.zeros((max_n_iter, n_sequences))
        start_cost = np.zeros(max_n_iter)
        transition_cost = np.zeros((max_n_iter, self.n_states))
        emission_cost = np.zeros((max_n_iter, self.n_states))

        ln_alpha_s = create_buffer_list(X_s, (self.n_states,), np_data_t)
        ln_beta_s = create_buffer_list(X_s, (self.n_states,), np_data_t)
        ln_gamma_s = create_buffer_list(X_s, (self.n_states,), np_data_t)
        ln_xi_s = create_buffer_list(X_s, (self.n_states, self.n_states), np_data_t)
        cdef data_t[:, :] ln_alpha, ln_beta, ln_gamma
        cdef data_t[:, :, :] ln_xi

        total_n_samples = np.concatenate(X_s, axis=0).shape[0]
        self.A_c = np.empty((total_n_samples, self.n_states, self.n_states), dtype=np_data_t)
        self.T_s = np.empty((n_sequences,), dtype=np.int)
        i = 0
        for p, X in enumerate(X_s):
            self.T_s[p] = i
            i += X_s[p].shape[0]

        B_s = create_buffer_list(X_s, (self.n_states,), np_data_t)
        cdef data_t[:, :] B

        memory_s = create_buffer_list(X_s, (self.n_states,), np_data_t)
        cdef data_t[:, :] memory

        # TODO: IF NOT SET BY USER
        n_hidden = n_features
        n_out = n_classes
        self.start_subnetwork = StartMLP(n_features, n_hidden, n_classes)
        self.transition_subnetworks = [
            TransitionMLP(n_features, n_hidden, n_classes) for i in range(self.n_states)]
        self.emission_subnetworks = [
            EmissionMLP(n_features, n_hidden, n_classes) for i in range(self.n_states)]

        for iteration in range(max_n_iter):
            print("Iteration number %i..." % iteration)
            for p in range(n_sequences):
                B = B_s[p]
                for i in range(self.n_states):
                    seq_length = X_s[p].shape[0]
                    # TODO: Made computation of B independent of the task (classification or regression)
                    R = np.log(self.emission_subnetworks[i].eval(X_s[p])[np.arange(seq_length), y_s[p]])
                    for k in range(seq_length):
                        B[k, i] = R[i]
                    R = np.log(self.transition_subnetworks[i].eval(X_s[p]))
                    for k in range(seq_length):
                        for j in range(self.n_states):
                            self.A_c[self.T_s[p]:self.T_s[p]+k, i, j] = R[k, j]
            
            # TODO: U and targets (y) must not be used after this line of code


            """ E-step"""
            lnP_s = np.empty(n_sequences)
            for p in range(n_sequences):
                lnP_s[p] = self.e_step(B_s[p], ln_alpha_s[p], ln_beta_s[p],
                    ln_gamma_s[p], ln_xi_s[p], p)
            lnP = np.sum(lnP_s)
            print(lnP)

            # TODO: CHECK CONVERGENCE THRESHOLD


            for p in range(n_sequences):
                B = B_s[p]
                ln_alpha = ln_alpha_s[p]
                memory = memory_s[p]
                R = np.squeeze(np.log(self.start_subnetwork.eval(X_s[p][0, :]))) # TODO: REMOVE U
                for i in range(self.n_states):
                    memory[0, i] = R[i]
                loglikelihood[iteration, p] = pyLogsum(np.asarray(ln_alpha[0, :]))
                seq_length = X_s[p].shape[0]
                start = self.T_s[p]
                with nogil:
                    for k in range(1, seq_length):
                        for i in range(self.n_states):
                            for l in range(self.n_states):
                                memory[k, l] = elnsum(
                                    memory[k, l],
                                    elnproduct(
                                        memory[k-1, i],
                                        self.A_c[start+k, i, l]
                                    )
                                )

            print("\tEnd of expectation step")
            """ M-Step """
            """
            pistate_cost[iteration] = piN.train(U, ln_gamma, n_epochs=parameters.pi_nepochs, 
                                           learning_rate=parameters.pi_learning_rate)
            for j in range(self.n_states):
                state_cost[iteration, j]  = N[j].train(U, ln_xi,
                            n_epochs=parameters.s_nepochs, learning_rate=parameters.s_learning_rate)
                output_cost[iteration, j] = O[j].train(U, targets_buf, memory,
                            n_epochs=parameters.o_nepochs, learning_rate=parameters.o_learning_rate)
            """