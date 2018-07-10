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

from archmm.ann.mlp import *
from archmm.ann.optimizers import *
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
    l = list()
    for sequence in X:
        buffer_shape = tuple([len(sequence)] + list(shape))
        l.append(np.empty(buffer_shape, dtype=dtype))
    return l


cdef class IOHMM(HMM):

    cdef bint is_classifier
    cdef int output_dim

    def __init__(self, n_states, arch='ergodic', is_classifier=True):
        HMM.__init__(self, n_states, arch=arch)
        self.is_classifier = is_classifier

    def fit(self, inputs, targets):
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
        cdef int i, j, t, k, l, p, iteration
        cdef int n_sequences = len(inputs)
        assert(n_sequences == len(targets))
        cdef cnp.int32_t[:] T = np.empty(n_sequences, dtype=np.int32)
        for p in range(n_sequences):
            T[p] = len(inputs[p])
        cdef cnp.ndarray U = typedListToPaddedTensor(inputs, np.asarray(T), is_3D=True, dtype=np_data_t)
        cdef cnp.int_t[:, :] targets_buf = typedListToPaddedTensor(targets, np.asarray(T), is_3D=False, dtype=np.int)
        cdef size_t m = U[0].shape[1]
        cdef size_t output_dim = targets[0].shape[1] if len(targets[0].shape) == 2 else 1
        cdef size_t r = n_classes if is_classifier else output_dim
        self.output_dim = r
        cdef object supervisor = Supervisor(targets, parameters.n_iterations, n_sequences, self.n_states)
        cdef data_t[:, :] loglikelihood = np.zeros((parameters.n_iterations, n_sequences), dtype=np_data_t)
        cdef data_t[:] pistate_cost = np.zeros(parameters.n_iterations, dtype=np_data_t)
        cdef data_t[:, :] state_cost = np.zeros((parameters.n_iterations, n), dtype=np_data_t)
        cdef data_t[:, :] output_cost = np.zeros((parameters.n_iterations, n), dtype=np_data_t)
        cdef ln_prob_t logalpha, logbeta, divider

        ln_alpha_s = create_buffer_list(inputs, (self.n_states,), np_data_t)
        ln_beta_s = create_buffer_list(inputs, (self.n_states,), np_data_t)
        ln_gamma_s = create_buffer_list(inputs, (self.n_states,), np_data_t)
        ln_xi_s = create_buffer_list(inputs, (self.n_states, self.n_states), np_data_t)
        cdef data_t[:, :] ln_alpha, ln_beta, ln_gamma
        cdef data_t[:, :, :] ln_xi

        A_s = create_buffer_list(inputs, (self.n_states, self.n_states), np_data_t)
        B_s = create_buffer_list(inputs, (self.n_states, self.output_dim), np_data_t)
        cdef data_t[:, :, :] A, B

        memory_s = create_buffer_list(inputs, (self.n_states,), np_data_t)
        cdef data_t[:, :] memory

        cdef data_t[:] ln_initial_probs
        cdef data_t[:] new_internal_state = np.empty(self.n_states, dtype=np_data_t)
        cdef cnp.float32_t[:] e_weights = np.ones(n_sequences, dtype=np.float32)
        cdef cnp.float32_t[:] t_weights = np.ones(2, dtype=np.float32)
        cdef cnp.float32_t[:] o_weights = np.ones(self.n_states, dtype=np.float32)

        cdef object N = list()
        cdef object O = list()
        for i in range(n):
            N.append(StateSubnetwork(i, m, parameters.s_nhidden, self.n_states, learning_rate = parameters.s_learning_rate,
                                     hidden_activation_function = parameters.s_activation, architecture = parameters.architecture))
        for i in range(n):
            O.append(OutputSubnetwork(i, m, parameters.o_nhidden, r, learning_rate = parameters.o_learning_rate,
                                      hidden_activation_function = parameters.o_activation))
        piN = PiStateSubnetwork(m, parameters.pi_nhidden, self.n_states, learning_rate = parameters.pi_learning_rate,
                                hidden_activation_function = parameters.pi_activation, architecture = parameters.architecture)


        


        for iteration in range(parameters.n_iterations):
            print("Iteration number %i..." % iteration)
            for j in range(n_sequences):
                A, B = A_s[j, :, :, :], B_s[j, :, :, :]
                for k in range(T[j]):
                    for i in range(self.n_states):
                        B[k, i, :] = np.log(O[i].computeOutput(U[j][k, :])[0])
                        A[k, i, :] = np.log(N[i].computeOutput(U[j][k, :])[0])
            """ Forward procedure """
            for j in range(n_sequences):
                A, B = A_s[j, :, :, :], B_s[j, :, :, :]
                ln_alpha = ln_alpha_s[j, :, :]
                memory = memory_s[j, :, :]
                memory[0, :] = np.log(piN.computeOutput(U[j][0, :]))
                with nogil:
                    for l in range(self.n_states):
                        ln_alpha[0, l] = elnproduct(B[0, l, <size_t>targets_buf[j][0]], memory[0, l])
                loglikelihood[iteration, j] = pyLogsum(np.asarray(ln_alpha[0, :]))
                for k in range(1, T[j]):
                    with nogil:
                        for i in range(self.n_states):
                            for l in range(self.n_states):
                                memory[k, l] = elnsum(
                                    memory[k, l],
                                    elnproduct(
                                        memory[k-1, i],
                                        A[k, i, l]
                                    )
                                )
                    loglikelihood[iteration, j] += pyLogsum(np.asarray(ln_alpha[k, :]))
                    with nogil:
                        for i in range(self.n_states):
                            logalpha = MINUS_INF
                            for l in range(self.n_states):
                                logalpha = elnsum(
                                    logalpha,
                                    elnproduct(ln_alpha[k-1, l], A[k, l, i])
                                )
                            ln_alpha[k, i] = elnproduct(logalpha, B[k, i, <size_t>targets_buf[j][k]])
                    loglikelihood[iteration, j] += pyLogsum(np.asarray(ln_alpha[k, :]))
            """ Backward procedure """
            for j in range(n_sequences):
                A, B = A_s[j, :, :, :], B_s[j, :, :, :]
                ln_beta = ln_beta_s[j, :, :]

                ln_beta[:, -1] = 0.0
                for k in range(T[j]-2, -1, -1):
                    for i in range(self.n_states):
                        logbeta = MINUS_INF
                        for l in range(self.n_states):
                            logbeta = elnsum(
                                logbeta,
                                elnproduct(
                                    A[k+1, i, l],
                                    elnproduct(
                                        B[k+1, l, <size_t>targets_buf[j][k + 1]],
                                        ln_beta[k+1, l]
                                    )
                                )
                            )
                        ln_beta[k, i] = logbeta
            """ Computation of state log-probabilities """
            for j in range(n_sequences):
                ln_alpha = ln_alpha_s[j, :, :]
                ln_beta = ln_beta_s[j, :, :]
                ln_gamma = ln_gamma_s[j, :, :]

                for k in range(T[j] - 1):
                    divider = MINUS_INF
                    for i in range(self.n_states):
                        ln_gamma[k, i] = elnproduct(ln_alpha[k, i], ln_beta[k, i])
                        divider = elnsum(divider, ln_gamma[k, i])
                    for i in range(self.n_states):
                        ln_gamma[k, i] = elnproduct(ln_gamma[k, i], -divider)
            """ Update model parameters """
            for j in range(n_sequences):
                ln_alpha = ln_alpha_s[j, :, :]
                ln_beta = ln_beta_s[j, :, :]
                ln_xi = ln_gamma_s[j, :, :, :]
                A, B = A_s[j, :, :, :], B_s[j, :, :, :]

                for k in range(T[j]-1):
                    divider = MINUS_INF
                    for i in range(self.n_states):
                        for l in range(self.n_states):
                            ln_xi[k, i, l] = elnproduct(
                                ln_alpha[k+1, l],
                                elnproduct(
                                    A[k+1, i, l],
                                    elnproduct(
                                        ln_beta[k+1, i],
                                        B[k+1, l, <size_t>targets_buf[j][k + 1]]
                                    )
                                )
                            )
                            divider = elnsum(divider, ln_xi[k, i, l])
                    for i in range(self.n_states):
                        for l in range(self.n_states):
                            ln_xi[k, i, l] = elnproduct(ln_xi[k, i, l], -divider)

            print("\tEnd of expectation step")
            """ M-Step """
            try:                    
                pistate_cost[iteration] = piN.train(U, ln_gamma, e_weights, n_epochs=parameters.pi_nepochs, 
                                               learning_rate=parameters.pi_learning_rate)
                for j in range(self.n_states):
                    state_cost[iteration, j]  = N[j].train(U, ln_xi, np.multiply(t_weights[0], e_weights),
                                n_epochs=parameters.s_nepochs, learning_rate=parameters.s_learning_rate)
                    output_cost[iteration, j] = O[j].train(U, targets_buf, memory, np.multiply(np.multiply(t_weights[1], e_weights), o_weights[j]),
                                n_epochs=parameters.o_nepochs, learning_rate=parameters.o_learning_rate)
                e_weights, t_weights, o_weights = supervisor.next(
                    np.asarray(loglikelihood[iteration]),
                    np.asarray(pistate_cost[iteration]), 
                    np.asarray(state_cost[iteration]), 
                    np.asarray(output_cost[iteration])
                )
                print("\tEnd of maximization step")
                print("\t\t-> Cost of the initial state subnetwork   : %f" % pistate_cost[iteration])
                print("\t\t-> Average cost of the state subnetworks  : %f" % np.asarray(state_cost[iteration]).mean())
                print("\t\t-> Average cost of the output subnetworks : %f" % np.asarray(output_cost[iteration]).mean())
                print("\t\t-> Average log-likelihood                 : %f" % np.asarray(loglikelihood[iteration]).mean())
            except MemoryError:
                print("\tMemory error : the maximization step had to be skipped")
                return loglikelihood[:iteration], pistate_cost[:iteration], state_cost[:iteration], output_cost[:iteration]
        return piN, N, O, loglikelihood, pistate_cost, state_cost, output_cost
