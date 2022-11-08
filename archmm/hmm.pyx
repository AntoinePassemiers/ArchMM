# -*- coding: utf-8 -*-
# hmm.pyx
# distutils: language=c
#
# Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.

from typing import List, Union, Any, Tuple, Sized, Collection, Type, Iterable

import cython
import numpy as np
cimport numpy as cnp
from archmm.distributions.base import BaseDistribution

from archmm.distributions.gaussian import MultivariateGaussian

cnp.import_array()

cimport libc.math


py_data_t = np.float

cdef data_t MINUS_INFINITY = <data_t>-np.inf
cdef data_t INFINITY = <data_t>np.inf


cdef inline int argmax(data_t[:] vec) nogil:
    cdef int i
    cdef int best_idx = 0
    cdef data_t best_val = -INFINITY
    for i in range(vec.shape[0]):
        if vec[i] > best_val:
            best_val = vec[i]
            best_idx = i
    return best_idx


cdef inline data_t sum_(data_t[:] vec) nogil:
    cdef int n = vec.shape[0]
    if n == 0:
        return 0.0
    cdef data_t s = 0.0
    for i in range(vec.shape[0]):
        s += vec[i]
    return s


cdef inline data_t log_sum_exp_1d(data_t[:] vec) nogil:
    cdef int n = vec.shape[0]
    if n == 0:
        return 0.0
    cdef data_t s = 0.0
    cdef int i = argmax(vec)
    cdef data_t offset = vec[i]
    if libc.math.isinf(offset):
        return MINUS_INFINITY
    for i in range(vec.shape[0]):
        s += libc.math.exp(vec[i] - offset)
    return libc.math.log(s) + offset


cdef inline data_t log_sum_exp(data_t x, data_t y) nogil:
    if x == MINUS_INFINITY:
        return y
    if y == MINUS_INFINITY:
        return x
    cdef data_t offset = libc.math.fmax(x, y)
    x = x - offset
    if x < -300:
        x = 0
    else:
        x = libc.math.exp(x)
    y = y - offset
    if y < -300:
        y = 0
    else:
        y = libc.math.exp(y)
    return offset + libc.math.log(x + y)


cdef class HMM:

    def __init__(self):

        # Hidden states
        self.states = []

        # Initial probabilities
        self.pi = np.ones(1, dtype=py_data_t)
        self.log_pi = np.log(self.pi).astype(py_data_t)

        # State transition probabilities
        self.a = np.ones((1, 1), dtype=py_data_t)
        self.log_a = np.log(self.a).astype(py_data_t)

    def init(self):

        # Initial probabilities
        self.pi = np.random.rand(self.n_states).astype(py_data_t)
        self.pi /= np.sum(self.pi)
        self.log_pi = np.log(self.pi).astype(py_data_t)

        # State transition probabilities
        self.a = np.random.rand(self.n_states, self.n_states).astype(py_data_t)
        self.a /= np.sum(self.a, axis=1)[:, np.newaxis]
        self.log_a = np.log(self.a).astype(py_data_t)

    def add_state(self, dist: BaseDistribution):
        self.states.append(dist)

    def add_states(self, dists: Iterable[BaseDistribution]):
        for dist in dists:
            self.add_state(dist)

    def emission_log_prob(self, sequence: np.ndarray, log_b: np.ndarray):
        for i in range(self.n_states):
            log_b[:, i] = self.states[i].log_pdf(sequence).astype(py_data_t)

    def emission_param_update(self, sequence: np.ndarray, gamma: np.ndarray):
        for i in range(self.n_states):
            self.states[i].param_update(sequence, gamma[:, i])

    def check_data(self, data: Union[np.ndarray, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, list):
            if len(data) == 0:
                return np.array([]), np.array([0])
            bounds = []
            start = 0
            for seq in data:
                bounds.append(start)
                start += len(seq)
            bounds.append(start)
            data = np.concatenate(data, axis=0)
        else:
            bounds = [0, len(data)]
        return data, np.asarray(bounds, dtype=int)

    cdef inline data_t forward_procedure(self, data_t[:, :] log_alpha, data_t[:, :] log_b, data_t[:] tmp) nogil:

        cdef int n = log_alpha.shape[0]
        cdef int n_states = log_alpha.shape[1]
        cdef int t, i, j

        # Forward procedure
        for i in range(n_states):
            log_alpha[0, i] = self.log_pi[i] + log_b[0, i]
        for t in range(1, n):
            for i in range(n_states):
                for j in range(n_states):
                    tmp[j] = log_alpha[t - 1, j] + self.log_a[j, i]
                log_alpha[t, i] = log_sum_exp_1d(tmp) + log_b[t, i]

        # Compute forward log-likelihood
        return log_sum_exp_1d(log_alpha[n - 1, :])

    cdef inline data_t backward_procedure(self, data_t[:, :] log_beta, data_t[:, :] log_b, data_t[:] tmp) nogil:

        cdef int n = log_beta.shape[0]
        cdef int n_states = log_beta.shape[1]
        cdef int t, i, j

        # Backward procedure
        for i in range(n_states):
            log_beta[n - 1, i] = 0
        for t in range(n - 2, -1, -1):
            for i in range(n_states):
                for j in range(n_states):
                    tmp[j] = log_beta[t + 1, j] + self.log_a[i, j] + log_b[t + 1, j]
                log_beta[t, i] = log_sum_exp_1d(tmp)

        # Compute backward log-likelihood
        for i in range(n_states):
            tmp[i] = self.log_pi[i] + log_b[0, i] + log_beta[0, i]
        return log_sum_exp_1d(tmp)

    def fit(self, data: Any, max_n_iter: int = 100):

        self.init()

        data, bounds_ = self.check_data(data)
        cdef int[:] bounds = np.asarray(bounds_, dtype=int)
        cdef int n_sequences = bounds.shape[0] - 1
        cdef int ns = len(data)
        cdef int n = 0
        cdef int n_states = self.n_states

        # Allocate memory
        cdef data_t[:, :] log_b_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_alpha_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_beta_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_gamma_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :, :] log_xi_ = np.zeros((ns, self.n_states, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_b
        cdef data_t[:, :] log_alpha
        cdef data_t[:, :] log_beta
        cdef data_t[:, :] log_gamma
        cdef data_t[:, :, :] log_xi
        cdef data_t[:] tmp = np.zeros(self.n_states, dtype=py_data_t)

        cdef int t, s, i, j, k, w, z, best_k
        cdef int start, end
        cdef data_t lse, forward_ll, backward_ll, log_numerator, log_denominator

        for _ in range(max_n_iter):

            self.emission_log_prob(data, np.asarray(log_b_, dtype=py_data_t))

            with nogil:

                for s in range(n_sequences):

                    start = bounds[s]
                    end = bounds[s + 1]
                    n = end - start
                    if n < 2:
                        continue
                    log_b = log_b_[start:end, :]
                    log_alpha = log_alpha_[start:end, :]
                    log_beta = log_beta_[start:end, :]
                    log_gamma = log_gamma_[start:end, :]
                    log_xi = log_xi_[start:end, :, :]

                    # Forward-backward algorithm
                    forward_ll = self.forward_procedure(log_alpha, log_b, tmp)
                    backward_ll = self.backward_procedure(log_beta, log_b, tmp)

                    # Compute occupation likelihood
                    for t in range(n):
                        for i in range(n_states):
                            log_gamma[t, i] = log_alpha[t, i] + log_beta[t, i]
                        lse = log_sum_exp_1d(log_gamma[t, :])
                        for i in range(n_states):
                            log_gamma[t, i] = log_gamma[t, i] - lse

                    # Compute transition likelihood
                    for t in range(n - 1):
                        for i in range(n_states):
                            for j in range(n_states):
                                log_xi[t, i, j] = log_alpha[t, i] + self.log_a[i, j] + log_beta[t + 1, j] + log_b[t + 1, j]
                        for i in range(n_states):
                            tmp[i] = log_sum_exp_1d(log_xi[t, i, :])
                        lse = log_sum_exp_1d(tmp)
                        for i in range(n_states):
                            for j in range(n_states):
                                log_xi[t, i, j] = log_xi    [t, i, j] - lse

                # Update initial state probabilities
                for i in range(n_states):
                    self.pi[i] = 0.0
                    for s in range(n_sequences):
                        start = bounds[s]
                        self.pi[i] += libc.math.exp(log_gamma[start, i])
                    self.pi[i] /= n_sequences
                    self.log_pi[i] = libc.math.log(self.pi[i])

                # Update state transition probabilities
                for i in range(n_states):

                    # Compute denominator
                    log_denominator = MINUS_INFINITY
                    for s in range(n_sequences):
                        start, end = bounds[s], bounds[s + 1]
                        n = end - start
                        if n < 2:
                            continue
                        log_gamma = log_gamma_[start:end, :]
                        for t in range(n - 1):
                            log_denominator = log_sum_exp(log_denominator, log_gamma[t, i])

                    for j in range(n_states):

                        # Compute numerator
                        log_numerator = MINUS_INFINITY
                        for s in range(n_sequences):
                            start, end = bounds[s], bounds[s + 1]
                            n = end - start
                            if n < 2:
                                continue
                            log_xi = log_xi_[start:end, :, :]
                            for t in range(log_xi.shape[0] - 1):
                                log_numerator = log_sum_exp(log_numerator, log_xi[t, i, j])
                        self.log_a[i, j] = log_numerator - log_denominator
                        self.a[i, j] = libc.math.exp(self.log_a[i, j])

            print(f'Likelihood: forward={forward_ll}, backward={backward_ll}')

            self.emission_param_update(data, np.exp(np.asarray(log_gamma_, dtype=py_data_t)))

    def decode(self, data: Any) -> np.ndarray:
        data, bounds_ = self.check_data(data)
        cdef int[:] bounds = np.asarray(bounds_, dtype=int)
        cdef int t, i, k, z, best_k
        cdef int n = len(data)
        cdef int n_states = self.n_states
        if n == 0:
            return np.asarray([])
        cdef data_t[:, :] log_b = np.zeros((n, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] t1 = np.zeros((n, self.n_states), dtype=py_data_t)
        cdef int[:, :] t2 = np.zeros((n, self.n_states), dtype=int)
        cdef data_t[:] tmp = np.zeros(self.n_states, dtype=py_data_t)
        cdef int[:] x = np.zeros(n, dtype=int)
        cdef data_t max_value, new_value

        self.emission_log_prob(data, log_b)

        with nogil:

            # Initial states
            for i in range(n_states):
                t1[0, i] = self.log_pi[i] + log_b[0, i]
                t2[0, i] = 0

            # State transitions
            for t in range(1, n):
                for i in range(n_states):
                    for k in range(n_states):
                        tmp[k] = t1[t - 1, k] + self.log_a[k, i] + log_b[t, i]
                    best_k = argmax(tmp)
                    t1[t, i] = tmp[best_k]
                    t2[t, i] = best_k

            # End state is the state that maximizes log-likelihood
            best_k = argmax(t1[n - 1, :])
            z = best_k

            # Traceback
            x[x.shape[0] - 1] = z
            for t in range(n - 1, 0, -1):
                z = t2[t, z]
                x[t - 1] = z

        return np.asarray(x)  # TODO: list of arrays

    def score(self, data: Any) -> float:
        data, bounds_ = self.check_data(data)
        cdef int[:] bounds = np.asarray(bounds_, dtype=int)
        cdef int n_sequences = bounds.shape[0] - 1
        cdef int ns = len(data)
        cdef int n = 0
        cdef int n_states = self.n_states
        cdef int s, start, end
        cdef data_t ll = 0

        # Allocate memory
        cdef data_t[:, :] log_b_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_alpha_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_b
        cdef data_t[:, :] log_alpha
        cdef data_t[:] tmp = np.zeros(self.n_states, dtype=py_data_t)

        self.emission_log_prob(data, np.asarray(log_b_, dtype=py_data_t))

        with nogil:
            for s in range(n_sequences):
                start = bounds[s]
                end = bounds[s + 1]
                n = end - start
                if n < 2:
                    continue
                log_b = log_b_[start:end, :]
                log_alpha = log_alpha_[start:end, :]
                ll += self.forward_procedure(log_alpha, log_b, tmp)
        return ll

    @property
    def n_states(self) -> int:
        return len(self.states)
