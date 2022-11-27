# -*- coding: utf-8 -*-
# hmm.pyx
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
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

import copy
from typing import Any, Iterable

cimport numpy as cnp
import numpy as np
cnp.import_array()

import scipy.special

from archmm.state import HiddenState
from archmm.utils import check_data

cimport libc.math


py_data_t = np.float

cdef data_t MINUS_INFINITY = <data_t>-np.inf
cdef data_t INFINITY = <data_t>np.inf


cdef inline int random_choice(data_t r, data_t[:] p) nogil:
    cdef int i
    cdef data_t cdf = 0.
    for i in range(p.shape[1]):
        cdf += p[i]
        if r <= cdf:
            return i
    return p.shape[0] - 1


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
        if vec[i] != MINUS_INFINITY:
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
        self.pi_mask = np.ones(1, dtype=np.uint8)

        # State transition probabilities
        self.a = np.ones((1, 1), dtype=py_data_t)
        self.log_a = np.log(self.a).astype(py_data_t)
        self.a_mask = np.ones((1, 1), dtype=np.uint8)

    def init(self):

        # Initial probabilities
        self.pi_mask = np.empty(self.n_states, dtype=bool)
        for i in range(self.n_states):
            self.pi_mask[i] = self.states[i].is_allowed_to_start()
        self.pi = (np.asarray(self.pi_mask) * np.random.rand(self.n_states)).astype(py_data_t)
        self.pi /= np.sum(self.pi)
        self.log_pi = np.log(self.pi).astype(py_data_t)

        # State transition probabilities
        self.a_mask = np.empty((self.n_states, self.n_states), dtype=bool)
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.a_mask[i, j] = self.states[i].can_transit_to(self.states[j])
        self.a = (np.asarray(self.a_mask) * np.random.rand(self.n_states, self.n_states)).astype(py_data_t)
        self.a /= np.sum(self.a, axis=1)[:, np.newaxis]
        self.a = np.nan_to_num(np.asarray(self.a, dtype=py_data_t), nan=0)
        self.log_a = np.log(self.a).astype(py_data_t)

    def add_state(self, state: HiddenState):
        self.states.append(state)

    def add_states(self, states: Iterable[HiddenState]):
        for state in states:
            self.add_state(state)

    def emission_log_prob(self, sequence: np.ndarray, log_b: np.ndarray):
        for i in range(self.n_states):
            log_b[:, i] = self.states[i].dist.log_pdf(sequence).astype(py_data_t)

    def emission_param_update(self, sequence: np.ndarray, gamma: np.ndarray):
        for i in range(self.n_states):
            self.states[i].dist.param_update(sequence, gamma[:, i])

    cdef inline data_t forward_procedure(self, data_t[:, :] log_alpha, data_t[:, :] log_b, data_t[:] tmp) nogil:

        cdef int n = log_alpha.shape[0]
        cdef int n_states = log_alpha.shape[1]
        cdef int t, i, j

        # Forward procedure
        for i in range(n_states):
            if self.pi_mask[i]:
                log_alpha[0, i] = self.log_pi[i] + log_b[0, i]
            else:
                log_alpha[0, i] = MINUS_INFINITY
        for t in range(1, n):
            for i in range(n_states):
                for j in range(n_states):
                    if self.a_mask[j, i]:
                        tmp[j] = log_alpha[t - 1, j] + self.log_a[j, i]
                    else:
                        tmp[j] = MINUS_INFINITY
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
                    if self.a_mask[i, j]:
                        tmp[j] = log_beta[t + 1, j] + self.log_a[i, j] + log_b[t + 1, j]
                    else:
                        tmp[j] = MINUS_INFINITY
                log_beta[t, i] = log_sum_exp_1d(tmp)

        # Compute backward log-likelihood
        for i in range(n_states):
            if self.pi_mask[i]:
                tmp[i] = self.log_pi[i] + log_b[0, i] + log_beta[0, i]
            else:
                tmp[i] = MINUS_INFINITY
        return log_sum_exp_1d(tmp)

    def fit(
            self,
            data: Any,
            max_n_iter: int = 100,
            eps: float = 1e-7,
            verbose: bool = True
    ):

        self.init()

        data, bounds_ = check_data(data)
        cdef cnp.int_t[:] bounds = np.asarray(bounds_, dtype=int)
        cdef int n_sequences = bounds.shape[0] - 1
        cdef int ns = len(data)
        cdef int n = 0
        cdef int n_states = self.n_states
        assert n_states > 0

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
        cdef list lls = []

        for iteration in range(max_n_iter):

            self.emission_log_prob(data, np.asarray(log_b_, dtype=py_data_t))

            with nogil:

                forward_ll = 0.
                backward_ll = 0.
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
                    forward_ll += self.forward_procedure(log_alpha, log_b, tmp)
                    backward_ll += self.backward_procedure(log_beta, log_b, tmp)

            # Compute occupation likelihood
            arr = (np.asarray(log_alpha_) + np.asarray(log_beta_)).astype(py_data_t)
            arr -= scipy.special.logsumexp(arr, axis=1)[:, np.newaxis]
            log_gamma_ = arr

            # Compute transition likelihood
            arr = np.asarray(log_alpha_)[:-1, :, np.newaxis] \
                  + np.asarray(self.log_a)[np.newaxis, :, :] \
                  + np.asarray(log_beta_)[1:, np.newaxis, :] \
                  + np.asarray(log_b_)[1:, np.newaxis, :]
            arr = arr.astype(py_data_t)
            arr -= scipy.special.logsumexp(arr, axis=(1, 2))[:, np.newaxis, np.newaxis]
            log_xi_ = arr

            # Update initial state probabilities
            idx = bounds_[:-1]
            self.pi = np.mean(np.exp(np.asarray(log_gamma_)[idx, :]), axis=0).astype(py_data_t)
            self.log_pi = np.log(np.asarray(self.pi)).astype(py_data_t)

            # TODO: optimize this
            with nogil:

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

            if verbose:
                print(f'Likelihood: forward={forward_ll}, backward={backward_ll}')
            lls.append(float(forward_ll))

            # Update parameters
            self.emission_param_update(data, np.exp(np.asarray(log_gamma_, dtype=py_data_t)))

            # Check convergence
            if len(lls) > 1:
                if abs((lls[iteration] - lls[iteration-1]) / lls[iteration-1]) < eps:
                    break

    def decode(self, data: Any) -> np.ndarray:
        data, bounds_ = check_data(data)
        cdef int[:] bounds = np.asarray(bounds_, dtype=int)
        cdef int t, i, k, z, best_k
        cdef int n = len(data)
        cdef int n_states = self.n_states
        if n == 0:
            return np.asarray([], dtype=int)
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
        data, bounds_ = check_data(data)
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

    def sample(self, length: int) -> np.ndarray:
        cdef int n = int(length)
        if n == 0:
            return np.asarray([])

        # Randomly generate the sequence of hidden states
        cdef int t
        cdef data_t[:] r = np.random.rand(n).astype(py_data_t)
        cdef data_t[:] p
        cdef cnp.int_t[:] states = np.zeros(n, dtype=int)
        with nogil:
            states[0] = random_choice(r[0], self.pi)
            for t in range(1, n):
                states[t] = random_choice(r[0], self.a[t-1, :])
        states_ = np.asarray(states)

        # Sample observations from hidden states
        data = np.empty((n, *self.states[0].dist.shape), dtype=float)  # TODO: distribution-specific dtype
        for i in range(self.n_states):
            idx = np.where(states_ == i)[0]
            if len(idx) > 0:
                data[idx, ...] = self.states[i].dist.sample(len(idx))
        return data

    def assert_almost_equal(self, other: 'HMM', precision: int = 6):
        # TODO: sort hidden states based on pairwise similarities
        np.testing.assert_array_almost_equal(self.log_pi, other.pi, decimal=precision)
        np.testing.assert_array_almost_equal(self.log_pi, other.a, decimal=precision)

    def __copy__(self) -> 'HMM':
        hmm = HMM()
        hmm.a = np.copy(np.asarray(self.a))
        hmm.log_a = np.copy(np.asarray(self.log_a))
        hmm.pi = np.copy(np.asarray(self.pi))
        hmm.log_pi = np.copy(np.asarray(self.log_pi))
        hmm.states = copy.copy(self.states)
        return hmm

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        hmm = self.__copy__()
        hmm.states = copy.deepcopy(self.states)
        return hmm

    @property
    def n_states(self) -> int:
        return len(self.states)
