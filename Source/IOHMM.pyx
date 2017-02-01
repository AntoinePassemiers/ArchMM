# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import numpy as np
from libc.stdio cimport *
from cython.parallel import parallel, prange
from threading import Thread

from archmm.ANN.MLP import *

include "Parallel.pyx"
include "structs.pyx"
include "Artifacts.pyx"


def IOHMMLinFit(inputs, targets = None, n_states = 2, dynamic_features = False, delta_window = 1, 
                is_classifier = True, n_classes = 2, parameters = None):
    cdef Py_ssize_t i, j, k, l, p, iter
    cdef size_t n_sequences = len(inputs)
    assert(n_sequences == len(targets))
    assert(parameters is not None)
    cdef cnp.ndarray T = np.empty(n_sequences, dtype = np.int32)
    for p in range(n_sequences):
        T[p] = len(inputs[p])
    cdef size_t T_max = T.max()
    cdef cnp.ndarray U = typedListToPaddedTensor(inputs, T, is_3D = True, dtype = np.float32)
    targets = typedListToPaddedTensor(targets, T, is_3D = False, dtype = np.int)
    cdef size_t m = U[0].shape[1]
    cdef size_t n = n_states
    cdef size_t output_dim = targets[0].shape[1] if len(targets[0].shape) == 2 else 1
    cdef size_t r = n_classes if is_classifier else output_dim
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

    cdef object supervisor = Supervisor(targets, parameters.n_iterations, n_sequences, n_states)
    cdef cnp.ndarray[cnp.double_t,ndim = 2] loglikelihood = np.zeros((parameters.n_iterations, n_sequences), dtype = np.double)
    cdef cnp.ndarray[cnp.double_t,ndim = 1] pistate_cost  = np.zeros(parameters.n_iterations, dtype = np.double)
    cdef cnp.ndarray[cnp.double_t,ndim = 2] state_cost    = np.zeros((parameters.n_iterations, n), dtype = np.double)
    cdef cnp.ndarray[cnp.double_t,ndim = 2] output_cost   = np.zeros((parameters.n_iterations, n), dtype = np.double)
    cdef cnp.ndarray[cnp.float32_t, ndim = 3] alpha = new3DVLMArray(n_sequences, n, T, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 3] beta  = new3DVLMArray(n_sequences, n, T, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 3] gamma = new3DVLMArray(n_sequences, n, T, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t,ndim = 4] xi    = new3DVLMArray(n_sequences, n, T, n, dtype = np.float32)
    cdef cnp.ndarray[cnp.float_t, ndim = 4] A = np.empty((n_sequences, T_max, n, n), dtype = np.float)
    cdef cnp.ndarray[cnp.float32_t, ndim = 2] initial_probs
    cdef cnp.ndarray[cnp.float32_t, ndim = 3] memory = np.empty((n_sequences, T_max, n), dtype = np.float32)
    cdef cnp.ndarray[cnp.float_t, ndim = 1] new_internal_state = np.empty(n, dtype = np.float)
    cdef cnp.ndarray[cnp.double_t,ndim = 4] B = new3DVLMArray(n_sequences, n, T, r, dtype = np.double)
    cdef cnp.ndarray is_mv = hasMissingValues(U, parameters.missing_value_sym)
    print(np.asarray(is_mv).shape)
    print(np.asarray(is_mv).shape)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] e_weights = np.ones(n_sequences, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] t_weights = np.ones(2, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] o_weights = np.ones(n_states, dtype = np.float32)
    
    for iter in range(parameters.n_iterations):
        print("Iteration number %i..." % iter)
        """ Forward procedure """
        for j in range(n_sequences):
            for k in range(T[j]):
                if not is_mv[j, k]:
                    for i in range(n):
                        B[j, i, k, :] = O[i].computeOutput(U[j][k, :])[0]
                        A[j, k, i, :] = N[i].computeOutput(U[j][k, :])[0]
        for j in range(n_sequences):
            if not is_mv[j, 0]:
                initial_probs = piN.computeOutput(U[j][0, :])
                memory[j, 0, :] = initial_probs    
                sequence_probs = np.multiply(B[j, :, 0, targets[j][0]], initial_probs)
                alpha[j, :, 0] = sequence_probs[:]
                loglikelihood[iter, j] = np.log(np.sum(sequence_probs))
            else:
                memory[j, 0, :] = sequence_probs = np.ones(n, np.float32)
                alpha[j, :, 0] = np.ones(n, dtype = np.float32)
                loglikelihood[iter, j] = 0.0
            for k in range(1, T[j]):
                new_internal_state[:] = 0
                for i in range(n):
                    new_internal_state[:] += memory[j, k - 1, i] * A[j, k, i, :]
                memory[j, k, :] = new_internal_state
                if not is_mv[j, k]:
                    for i in range(n):
                        alpha[j, i, k] = 0
                        for l in range(n):
                            alpha[j, i, k] += alpha[j, l, k - 1] * A[j, k, l, i]
                        alpha[j, i, k] *= B[j, i, k, targets[j][k]]
                    loglikelihood[iter, j] += np.log(np.sum(alpha[j, :, k]))
                else: # If value is missing
                    for i in range(n):
                        alpha[j, i, k] = 0
                        for l in range(n):
                            alpha[j, i, k] += alpha[j, l, k - 1]
        """ Backward procedure """
        for j in range(n_sequences):
            beta[j, :, -1] = 1
            for k in range(T[j] - 2, -1, -1):
                if not is_mv[j, k]:
                    for i in range(n):
                        beta[j, i, k] = 0
                        for l in range(n):
                            beta[j, i, k] += beta[j, l, k + 1] * A[j, k + 1, i, l] * B[j, l, k, targets[j][k]]
                else: # If value is missing
                    for i in range(n):
                        beta[j, i, k] = 0
                        for l in range(n):
                            beta[j, i, k] += beta[j, l, k + 1]
        """ Forward-Backward xi computation """
        for j in range(n_sequences):
            """
            for k in range(T[j]):
                temp = np.multiply(alpha[j, :, k], beta[j, :, k])
                gamma[j, :, k] = np.divide(temp, temp.sum())
            """
            temp = np.multiply(alpha[j, :, 0], beta[j, :, 0])
            gamma[j, :, 0] = temp / temp.sum()
        for j in range(n_sequences):
            denominator = np.sum(alpha[j, :, -1])
            xi[j, :, 0, :] = 0 # TODO ???
            for k in range(T[j] - 1):
                for i in range(n):
                    for l in range(n):
                        # xi[j, :, k, :] = np.multiply(A, alpha[j, :, k] * np.multiply(B[j, :, k + 1, targets[j][k + 1]], beta[j, :, k + 1]))
                        xi[j, i, k + 1, l] = beta[j, i, k + 1] * alpha[j, l, k] * A[j, k + 1, i, l] / denominator
        gamma = nan_to_num(gamma)
        xi = nan_to_num(xi)
        memory = nan_to_num(memory)
        print("\tEnd of expectation step")
        print("\t\t-> Average log-likelihood : %f" % loglikelihood[iter].mean())
        """ M-Step """
        """ 
        memory[j, k, i] is the probability that the current state for the sequence j at time k is i 
        xi[j, i, k, l] measures the expectation that, for the sequence j, the current state at
        time k is i and the current state at time k - 1 is l 
        """
        try:
            pistate_cost[iter] = piN.train(U, gamma, e_weights, n_epochs = parameters.pi_nepochs, 
                                           learning_rate = parameters.pi_learning_rate)
            for j in range(n):
                state_cost[iter, j]  = N[j].train(U, xi, t_weights[0] * e_weights, 
                            is_mv, n_epochs = parameters.s_nepochs, learning_rate = parameters.s_learning_rate)
                output_cost[iter, j] = O[j].train(U, targets, memory, t_weights[1] * e_weights * o_weights[j], 
                            is_mv, n_epochs = parameters.o_nepochs, learning_rate = parameters.o_learning_rate)
            e_weights, t_weights, o_weights = supervisor.next(loglikelihood[iter], 
                    pistate_cost[iter], state_cost[iter], output_cost[iter])
            print("\tEnd of maximization step")
            print("\t\t-> Cost of the initial state subnetwork : %f" % pistate_cost[iter])
            print("\t\t-> Average cost of the state subnetworks : %f" % state_cost[iter].mean())
            print("\t\t-> Average cost of the output subnetworks : %f" % output_cost[iter].mean())
        except MemoryError:
            print("\tMemory error : the maximization step had to be skipped")
            return piN, N, O, loglikelihood[:iter], pistate_cost[:iter], state_cost[:iter], output_cost[:iter], supervisor.weights
    return piN, N, O, loglikelihood, pistate_cost, state_cost, output_cost, supervisor.weights

def logsum(vec):
    return np.log(np.sum(np.exp(vec)))

def binaryLogsum(x, y):
    return np.log(np.exp(x) + np.exp(y))

def IOHMMLogFit(inputs, targets = None, n_states = 2, dynamic_features = False, delta_window = 1, 
                is_classifier = True, n_classes = 2, parameters = None):
    cdef Py_ssize_t i, j, k, l, p, iter
    cdef size_t n_sequences = len(inputs)
    assert(n_sequences == len(targets))
    assert(parameters is not None)
    cdef cnp.ndarray T = np.empty(n_sequences, dtype = np.int32)
    for p in range(n_sequences):
        T[p] = len(inputs[p])
    cdef size_t T_max = T.max()
    cdef cnp.ndarray U = typedListToPaddedTensor(inputs, T, is_3D = True, dtype = np.float32)
    targets = typedListToPaddedTensor(targets, T, is_3D = False, dtype = np.int)
    cdef size_t m = U[0].shape[1]
    cdef size_t n = n_states
    cdef size_t output_dim = targets[0].shape[1] if len(targets[0].shape) == 2 else 1
    cdef size_t r = n_classes if is_classifier else output_dim
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

    cdef object supervisor = Supervisor(targets, parameters.n_iterations, n_sequences, n_states)
    cdef cnp.ndarray[cnp.double_t,ndim = 2] loglikelihood = np.zeros((parameters.n_iterations, n_sequences), dtype = np.double)
    cdef cnp.ndarray[cnp.double_t,ndim = 1] pistate_cost  = np.zeros(parameters.n_iterations, dtype = np.double)
    cdef cnp.ndarray[cnp.double_t,ndim = 2] state_cost    = np.zeros((parameters.n_iterations, n), dtype = np.double)
    cdef cnp.ndarray[cnp.double_t,ndim = 2] output_cost   = np.zeros((parameters.n_iterations, n), dtype = np.double)
    cdef cnp.ndarray[cnp.float32_t, ndim = 3] alpha = new3DVLMArray(n_sequences, n, T, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 3] beta  = new3DVLMArray(n_sequences, n, T, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 3] gamma = new3DVLMArray(n_sequences, n, T, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t,ndim = 4] xi    = new3DVLMArray(n_sequences, n, T, n, dtype = np.float32)
    cdef cnp.ndarray[cnp.float_t, ndim = 4] A = np.empty((n_sequences, T_max, n, n), dtype = np.float)
    cdef cnp.ndarray[cnp.float32_t, ndim = 2] initial_probs
    cdef cnp.ndarray[cnp.float32_t, ndim = 3] memory = np.empty((n_sequences, T_max, n), dtype = np.float32)
    cdef cnp.ndarray[cnp.float_t, ndim = 1] new_internal_state = np.empty(n, dtype = np.float)
    cdef cnp.ndarray[cnp.double_t, ndim = 4] B = np.empty((n_sequences, n, T.max(), r), dtype = np.double)
    cdef cnp.ndarray is_mv = hasMissingValues(U, parameters.missing_value_sym)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] e_weights = np.ones(n_sequences, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] t_weights = np.ones(2, dtype = np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] o_weights = np.ones(n_states, dtype = np.float32)
    
    for iter in range(parameters.n_iterations):
        print("Iteration number %i..." % iter)
        """ Forward procedure """
        for j in range(n_sequences):
            for k in range(T[j]):
                if not is_mv[j, k]:
                    for i in range(n):
                        B[j, i, k, :] = O[i].computeOutput(U[j][k, :])[0]
                        A[j, k, i, :] = N[i].computeOutput(U[j][k, :])[0]
        for j in range(n_sequences):
            if not is_mv[j, 0]:
                initial_probs = piN.computeOutput(U[j][0, :])
                memory[j, 0, :] = initial_probs    
                sequence_probs = np.multiply(B[j, :, 0, targets[j][0]], initial_probs)
                alpha[j, :, 0] = sequence_probs[:]
                loglikelihood[iter, j] = np.log(np.sum(sequence_probs))
            else:
                memory[j, 0, :] = sequence_probs = np.ones(n, np.float32)
                alpha[j, :, 0] = np.ones(n, dtype = np.float32)
                loglikelihood[iter, j] = 0.0
            for k in range(1, T[j]):
                new_internal_state[:] = 0
                for i in range(n):
                    new_internal_state[:] += memory[j, k - 1, i] * A[j, k, i, :]
                memory[j, k, :] = new_internal_state
                if not is_mv[j, k]:
                    for i in range(n):
                        alpha[j, i, k] = 0
                        for l in range(n):
                            alpha[j, i, k] += alpha[j, l, k - 1] * A[j, k, l, i]
                        alpha[j, i, k] *= B[j, i, k, targets[j][k]]
                    loglikelihood[iter, j] += np.log(np.sum(alpha[j, :, k]))
                else: # If value is missing
                    for i in range(n):
                        alpha[j, i, k] = 0
                        for l in range(n):
                            alpha[j, i, k] += alpha[j, l, k - 1]
        """ Backward procedure """
        for j in range(n_sequences):
            beta[j, :, -1] = 1
            for k in range(T[j] - 2, -1, -1):
                if not is_mv[j, k]:
                    for i in range(n):
                        beta[j, i, k] = 0
                        for l in range(n):
                            beta[j, i, k] += beta[j, l, k + 1] * A[j, k + 1, i, l] * B[j, l, k, targets[j][k]]
                else: # If value is missing
                    for i in range(n):
                        beta[j, i, k] = 0
                        for l in range(n):
                            beta[j, i, k] += beta[j, l, k + 1]
        """ Forward-Backward xi computation """
        for j in range(n_sequences):
            """
            for k in range(T[j]):
                temp = np.multiply(alpha[j, :, k], beta[j, :, k])
                gamma[j, :, k] = np.divide(temp, temp.sum())
            """
            temp = np.multiply(alpha[j, :, 0], beta[j, :, 0])
            gamma[j, :, 0] = temp / temp.sum()
        for j in range(n_sequences):
            denominator = np.sum(alpha[j, :, -1])
            xi[j, :, 0, :] = 0 # TODO ???
            for k in range(T[j] - 1):
                for i in range(n):
                    for l in range(n):
                        # xi[j, :, k, :] = np.multiply(A, alpha[j, :, k] * np.multiply(B[j, :, k + 1, targets[j][k + 1]], beta[j, :, k + 1]))
                        xi[j, i, k + 1, l] = beta[j, i, k + 1] * alpha[j, l, k] * A[j, k + 1, i, l] / denominator
        gamma = nan_to_num(gamma)
        xi = nan_to_num(xi)
        memory = nan_to_num(memory)
        print("\tEnd of expectation step")
        print("\t\t-> Average log-likelihood : %f" % loglikelihood[iter].mean())
        """ M-Step """
        """ 
        memory[j, k, i] is the probability that the current state for the sequence j at time k is i 
        xi[j, i, k, l] measures the expectation that, for the sequence j, the current state at
        time k is i and the current state at time k - 1 is l 
        """
        pi_threads, n_threads, o_threads = list(), list(), list()
        try:                    
            pi_threads.append(train_pi_network(piN, U, gamma, e_weights, n_epochs = parameters.pi_nepochs, 
                                           learning_rate = parameters.pi_learning_rate))
            pi_threads[0].start()
            for j in range(n):
                n_threads.append(train_state_network(N[j], U, xi, t_weights[0] * e_weights, 
                            is_mv, n_epochs = parameters.s_nepochs, learning_rate = parameters.s_learning_rate))
                n_threads[j].start()
                o_threads.append(train_output_network(O[j], U, targets, memory, t_weights[1] * e_weights * o_weights[j], 
                            is_mv, n_epochs = parameters.o_nepochs, learning_rate = parameters.o_learning_rate))
                o_threads[j].start()
    
            pistate_cost[iter] = pi_threads[0].join()
            for j in range(n):
                state_cost[iter, j]  = n_threads[j].join()
                output_cost[iter, j] = o_threads[j].join()
                
                
            e_weights, t_weights, o_weights = supervisor.next(loglikelihood[iter], 
                    pistate_cost[iter], state_cost[iter], output_cost[iter])
            print("\tEnd of maximization step")
            print("\t\t-> Cost of the initial state subnetwork : %f" % pistate_cost[iter])
            print("\t\t-> Average cost of the state subnetworks : %f" % state_cost[iter].mean())
            print("\t\t-> Average cost of the output subnetworks : %f" % output_cost[iter].mean())
        except MemoryError:
            print("\tMemory error : the maximization step had to be skipped")
            return loglikelihood[:iter], pistate_cost[:iter], state_cost[:iter], output_cost[:iter]
    return piN, N, O, loglikelihood, pistate_cost, state_cost, output_cost, supervisor.weights
