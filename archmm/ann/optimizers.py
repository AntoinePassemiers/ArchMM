# -*- coding: utf-8 -*-
# optimizers.py : Optimizer for stable convergence during stochastic gradient descent
# author: Antoine Passemiers

import numpy as np

class Supervisor:
    def __init__(self, targets, n_iterations, n_examples, n_states):
        self.n_examples = n_examples
        self.n_states = n_states
        self.n_iterations = n_iterations
        self.history = np.empty((n_iterations + 1, n_examples), dtype = np.float32)
        self.history[0, :] = - np.inf
        self.weights = np.ones((n_iterations + 1, n_examples), dtype = np.float32)
        self.labels = np.empty(n_examples, dtype = np.int32)
        for i in range(n_examples):
            self.labels[i] = targets[i][-1]
        self.current_iter = 0
        self.hardworker = 0
    def next(self, loglikelihoods, pi_state_costs, state_costs, output_costs):
        self.hardworker = output_costs.argmax()
        self.current_iter += 1
        self.history[self.current_iter] = loglikelihoods
        signs = np.sign(loglikelihoods - self.history[self.current_iter - 1])
        signs[np.isnan(signs)] = 0
        # indexes = (signs <= 0)
        indexes = np.less(loglikelihoods, 0)
        decay = float(self.n_iterations - self.current_iter) / float(self.n_iterations)
        self.weights[self.current_iter, indexes] = 3 * decay * np.sqrt(self.weights[self.current_iter - 1, indexes]) + 1
        self.weights[self.current_iter, loglikelihoods > 0] = 0
        t_weights = np.ones(2, dtype = np.float32)
        o_weights = np.ones(self.n_states, dtype = np.float32)
        # o_weights[self.hardworker] = 1.0
        return self.weights[self.current_iter], t_weights, o_weights