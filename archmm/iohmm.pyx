# -*- coding: utf-8 -*-
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
from archmm.dbn cimport *
from archmm.structs cimport *

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

def IOHMMLogFit(inputs, targets = None, n_states = 2, dynamic_features = False, delta_window = 1, 
                is_classifier = True, n_classes = 2, parameters = None):
    cdef Py_ssize_t i, j, k, l, p, iter
    cdef size_t n_sequences = len(inputs)
    assert(n_sequences == len(targets))
    assert(parameters is not None)
    cdef cnp.int32_t[:] T = np.empty(n_sequences, dtype = np.int32)
    for p in range(n_sequences):
        T[p] = len(inputs[p])
    cdef size_t T_max = np.asarray(T).max()
    cdef cnp.ndarray U = typedListToPaddedTensor(inputs, np.asarray(T), is_3D = True, dtype = np.float32)
    cdef cnp.int_t[:, :] targets_buf = typedListToPaddedTensor(targets, np.asarray(T), is_3D = False, dtype = np.int)
    cdef size_t m = U[0].shape[1]
    cdef size_t n = n_states
    cdef size_t output_dim = targets[0].shape[1] if len(targets[0].shape) == 2 else 1
    cdef size_t r = n_classes if is_classifier else output_dim
    cdef object supervisor = Supervisor(targets, parameters.n_iterations, n_sequences, n_states)
    cdef cnp.double_t[:, :] loglikelihood = np.zeros((parameters.n_iterations, n_sequences), dtype = np.double)
    cdef cnp.double_t[:] pistate_cost     = np.zeros(parameters.n_iterations, dtype = np.double)
    cdef cnp.double_t[:, :] state_cost    = np.zeros((parameters.n_iterations, n), dtype = np.double)
    cdef cnp.double_t[:, :] output_cost   = np.zeros((parameters.n_iterations, n), dtype = np.double)
    cdef ln_prob_t logalpha, logbeta, divider
    cdef ln_prob_t[:, :, :] ln_alpha = new3DVLMArray(n_sequences, n, np.asarray(T), dtype = np.float32)
    cdef ln_prob_t[:, :, :] ln_beta  = new3DVLMArray(n_sequences, n, np.asarray(T), dtype = np.float32)
    cdef ln_prob_t[:, :, :] ln_gamma = new3DVLMArray(n_sequences, n, np.asarray(T), dtype = np.float32)
    cdef ln_prob_t[:, :, :, :] ln_xi = new3DVLMArray(n_sequences, n, np.asarray(T), n, dtype = np.float32)
    cdef cnp.ndarray[cnp.float_t, ndim = 4] A = np.empty((n_sequences, T_max, n, n), dtype = np.float)
    cdef ln_prob_t[:] ln_initial_probs
    cdef cnp.ndarray[cnp.float32_t, ndim = 3] memory = np.empty((n_sequences, T_max, n), dtype = np.float32)
    cdef cnp.ndarray[cnp.float_t, ndim = 1] new_internal_state = np.empty(n, dtype = np.float)
    cdef cnp.ndarray[cnp.double_t,ndim = 4] B = new3DVLMArray(n_sequences, n, np.asarray(T), r, dtype = np.double)
    cdef cnp.ndarray is_mv = hasMissingValues(U, parameters.missing_value_sym)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] e_weights = np.ones(n_sequences, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] t_weights = np.ones(2, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] o_weights = np.ones(n_states, dtype = np.float32)
    cdef object N = list()
    cdef object O = list()
    for i in range(n):
        N.append(StateSubnetwork(i, m, parameters.s_nhidden, n, learning_rate = parameters.s_learning_rate,
                                 hidden_activation_function = parameters.s_activation, architecture = parameters.architecture))
    for i in range(n):
        O.append(OutputSubnetwork(i, m, parameters.o_nhidden, r, learning_rate = parameters.o_learning_rate,
                                  hidden_activation_function = parameters.o_activation))
    piN = PiStateSubnetwork(m, parameters.pi_nhidden, n, learning_rate = parameters.pi_learning_rate,
                            hidden_activation_function = parameters.pi_activation, architecture = parameters.architecture)

    for iter in range(parameters.n_iterations):
        print("Iteration number %i..." % iter)
        """ Forward procedure """
        for j in range(n_sequences):
            for k in range(T[j]):
                if not is_mv[j, k]:
                    for i in range(n):
                        B[j, i, k, :] = np.log(O[i].computeOutput(U[j][k, :])[0])
                        A[j, k, i, :] = np.log(N[i].computeOutput(U[j][k, :])[0])
        for j in range(n_sequences):
            if not is_mv[j, 0]:
                memory[j, 0, :] = np.log(piN.computeOutput(U[j][0, :]))
                with nogil:
                    for l in range(n):
                        ln_alpha[j, l, 0] = elnproduct(B[j, l, 0, <size_t>targets_buf[j][0]], memory[j, 0, l])
                loglikelihood[iter, j] = pyLogsum(np.asarray(ln_alpha[j, :, 0]))
            else:
                memory[j, 0, :] = np.ones(n, np.float32)
                ln_alpha[j, :, 0] = 0
            for k in range(1, T[j]):
                with nogil:
                    for i in range(n):
                        for l in range(n):
                            memory[j, k, l] = elnsum(
                                memory[j, k, l],
                                elnproduct(
                                    memory[j, k - 1, i],
                                    A[j, k, i, l]
                                )
                            )
                loglikelihood[iter, j] += pyLogsum(np.asarray(ln_alpha[j, :, k]))
                if not is_mv[j, k]:
                    with nogil:
                        for i in range(n):
                            logalpha = MINUS_INF
                            for l in range(n):
                                logalpha = elnsum(
                                    logalpha,
                                    elnproduct(ln_alpha[j, l, k - 1], A[j, k, l, i])
                                )
                            ln_alpha[j, i, k] = elnproduct(logalpha, B[j, i, k, <size_t>targets_buf[j][k]])
                else: # If value is missing
                    with nogil:
                        for i in range(n):
                            ln_alpha[j, i, k] = MINUS_INF
                            for l in range(n):
                                ln_alpha[j, i, k] = elnsum(ln_alpha[j, l, k - 1], ln_alpha[j, i, k])
                loglikelihood[iter, j] += pyLogsum(np.asarray(ln_alpha[j, :, k]))
        """ Backward procedure """
        for j in range(n_sequences):
            ln_beta[j, :, -1] = 0.0
            for k in range(T[j] - 2, -1, -1):
                if not is_mv[j, k]:
                    for i in range(n):
                        logbeta = MINUS_INF
                        for l in range(n):
                            logbeta = elnsum(
                                logbeta,
                                elnproduct(
                                    A[j, k + 1, i, l],
                                    elnproduct(
                                        B[j, l, k + 1, <size_t>targets_buf[j][k + 1]],
                                        ln_beta[j, l, k + 1]
                                    )
                                )
                            )
                        ln_beta[j, i, k] = logbeta
                else: # If value is missing
                    for i in range(n):
                        ln_beta[j, i, k] = MINUS_INF
                        for l in range(n):
                            ln_beta[j, i, k] = elnsum(ln_beta[j, i, k], ln_beta[j, l, k + 1])
        """ Computation of state log-probabilities """
        for j in range(n_sequences):
            for k in range(T[j] - 1):
                divider = MINUS_INF
                for i in range(n):
                    ln_gamma[j, i, k] = elnproduct(ln_alpha[j, i, k], ln_beta[j, i, k])
                    divider = elnsum(divider, ln_gamma[j, i, k])
                for i in range(n):
                    ln_gamma[j, i, k] = elnproduct(ln_gamma[j, i, k], -divider)
        """ Update model parameters """
        for j in range(n_sequences):
            for k in range(T[j] - 1):
                divider = MINUS_INF
                for i in range(n):
                    for l in range(n):
                        ln_xi[j, i, k, l] = elnproduct(
                            ln_alpha[j, l, k + 1],
                            elnproduct(
                                A[j, k + 1, i, l],
                                elnproduct(
                                    ln_beta[j, i, k + 1],
                                    B[j, l, k + 1, <size_t>targets_buf[j][k + 1]]
                                )
                            )
                        )
                        divider = elnsum(divider, ln_xi[j, i, k, l])
                for i in range(n):
                    for l in range(n):
                        ln_xi[j, i, k, l] = elnproduct(ln_xi[j, i, k, l], -divider)

        print("\tEnd of expectation step")
        """ M-Step """
        try:                    
            pistate_cost[iter] = piN.train(U, ln_gamma, e_weights, n_epochs = parameters.pi_nepochs, 
                                           learning_rate = parameters.pi_learning_rate)
            for j in range(n):
                state_cost[iter, j]  = N[j].train(U, ln_xi, t_weights[0] * e_weights, 
                            is_mv, n_epochs = parameters.s_nepochs, learning_rate = parameters.s_learning_rate)
                output_cost[iter, j] = O[j].train(U, targets_buf, memory, t_weights[1] * e_weights * o_weights[j], 
                            is_mv, n_epochs = parameters.o_nepochs, learning_rate = parameters.o_learning_rate)
            e_weights, t_weights, o_weights = supervisor.next(
                np.asarray(loglikelihood[iter]),
                np.asarray(pistate_cost[iter]), 
                np.asarray(state_cost[iter]), 
                np.asarray(output_cost[iter])
            )
            print("\tEnd of maximization step")
            print("\t\t-> Cost of the initial state subnetwork   : %f" % pistate_cost[iter])
            print("\t\t-> Average cost of the state subnetworks  : %f" % np.asarray(state_cost[iter]).mean())
            print("\t\t-> Average cost of the output subnetworks : %f" % np.asarray(output_cost[iter]).mean())
            print("\t\t-> Average log-likelihood                 : %f" % np.asarray(loglikelihood[iter]).mean())
        except MemoryError:
            print("\tMemory error : the maximization step had to be skipped")
            return loglikelihood[:iter], pistate_cost[:iter], state_cost[:iter], output_cost[:iter]
    return piN, N, O, loglikelihood, pistate_cost, state_cost, output_cost, None
