# -*- coding: utf-8 -*-
# iohmm.pyx
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

from typing import Any

import numpy as np
cimport numpy as cnp
cnp.import_array()

import scipy.special

from archmm.hmm cimport data_t, log_sum_exp_1d
from archmm.hmm import py_data_t, HMM

from archmm.utils import check_data


cdef class IOHMM:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_model = None
        self.transition_models = []
        self.emission_models = []

    def initial_log_prob(self, sequence: np.ndarray, log_pi: np.ndarray):
        log_pi[:, :] = self.initial_model.log_prob(sequence).astype(py_data_t)

    def transition_log_prob(self, sequence: np.ndarray, log_a: np.ndarray):
        for i in range(self.n_states):
            log_a[:, i, :] = self.transition_models[i].log_prob(sequence).astype(py_data_t)

    def emission_log_prob(self, X: np.ndarray, Y: np.ndarray, log_b: np.ndarray):
        for i in range(self.n_states):
            log_b[:, i] = self.emission_models[i].log_pdf(X, Y).astype(py_data_t)

    cdef inline data_t forward_procedure(
            self,
            data_t[:] log_pi,
            data_t[:, :] log_alpha,
            data_t[:, :, :] log_a,
            data_t[:, :] log_b,
            data_t[:] tmp
    ) nogil:

        cdef int n = log_alpha.shape[0]
        cdef int n_states = log_alpha.shape[1]
        cdef int t, i, j

        # Forward procedure
        for i in range(n_states):
            log_alpha[0, i] = log_pi[i] + log_b[0, i]
        for t in range(1, n):
            for i in range(n_states):
                for j in range(n_states):
                    tmp[j] = log_alpha[t - 1, j] + log_a[t, j, i]
                log_alpha[t, i] = log_sum_exp_1d(tmp) + log_b[t, i]

        # Compute forward log-likelihood
        return log_sum_exp_1d(log_alpha[n - 1, :])

    cdef inline data_t backward_procedure(
            self,
            data_t[:] log_pi,
            data_t[:, :] log_beta,
            data_t[:, :, :] log_a,
            data_t[:, :] log_b,
            data_t[:] tmp
    ) nogil:

        cdef int n = log_beta.shape[0]
        cdef int n_states = log_beta.shape[1]
        cdef int t, i, j

        # Backward procedure
        for i in range(n_states):
            log_beta[n - 1, i] = 0
        for t in range(n - 2, -1, -1):
            for i in range(n_states):
                for j in range(n_states):
                    tmp[j] = log_beta[t + 1, j] + log_a[t, i, j] + log_b[t + 1, j]
                log_beta[t, i] = log_sum_exp_1d(tmp)

        # Compute backward log-likelihood
        for i in range(n_states):
            tmp[i] = log_pi[i] + log_b[0, i] + log_beta[0, i]
        return log_sum_exp_1d(tmp)

    def fit(self, X: Any, Y: Any, max_n_iter: int = 100):

        X, bounds_ = check_data(X)
        Y, _ = check_data(Y)
        cdef int[:] bounds = np.asarray(bounds_, dtype=int)
        cdef int n_sequences = bounds.shape[0] - 1
        cdef int ns = len(X)
        cdef int n = 0
        cdef int n_states = self.n_states

        # Allocate memory
        cdef data_t[:, :] log_pi_ = np.zeros((n_sequences, self.n_states), dtype=py_data_t)
        cdef data_t[:, :, :] log_a_ = np.zeros((ns, self.n_states, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_b_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_alpha_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_beta_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :] log_gamma_ = np.zeros((ns, self.n_states), dtype=py_data_t)
        cdef data_t[:, :, :] log_xi_ = np.zeros((ns, self.n_states, self.n_states), dtype=py_data_t)
        cdef data_t[:] log_pi
        cdef data_t[:, :] log_a, log_b, log_alpha, log_beta, log_gamma
        cdef data_t[:, :, :] log_xi
        cdef data_t[:] tmp = np.zeros(self.n_states, dtype=py_data_t)

        cdef int t, s, i, j, k, w, z, best_k
        cdef int start, end
        cdef data_t lse, forward_ll, backward_ll, log_numerator, log_denominator

        for _ in range(max_n_iter):

            self.initial_log_prob(X[bounds_[:-1]], np.asarray(log_pi_, dtype=py_data_t))
            self.transition_log_prob(X, np.asarray(log_a_, dtype=py_data_t))
            self.emission_log_prob(X, Y, np.asarray(log_b_, dtype=py_data_t))

            with nogil:

                for s in range(n_sequences):

                    start = bounds[s]
                    end = bounds[s + 1]
                    n = end - start
                    if n < 2:
                        continue
                    log_pi = log_pi_[start, :]
                    log_a = log_a_[start:end, :]
                    log_b = log_b_[start:end, :]
                    log_alpha = log_alpha_[start:end, :]
                    log_beta = log_beta_[start:end, :]
                    log_gamma = log_gamma_[start:end, :]
                    log_xi = log_xi_[start:end, :, :]

                    # Forward-backward algorithm
                    forward_ll = self.forward_procedure(log_pi, log_alpha, log_b, tmp)
                    backward_ll = self.backward_procedure(log_pi, log_beta, log_b, tmp)

            # Compute occupation likelihood
            arr = (np.asarray(log_alpha_) + np.asarray(log_beta_)).astype(py_data_t)
            arr -= scipy.special.logsumexp(arr, axis=1)[:, np.newaxis]
            log_gamma_ = arr

            # Compute transition likelihood
            arr = np.asarray(log_alpha_)[:-1, :, np.newaxis] \
                  + np.asarray(log_a_)[1:, :, :] \
                  + np.asarray(log_beta_)[1:, np.newaxis, :] \
                  + np.asarray(log_b_)[1:, np.newaxis, :]
            arr = arr.astype(py_data_t)
            arr -= scipy.special.logsumexp(arr, axis=(1, 2))[:, np.newaxis, np.newaxis]
            log_xi_ = arr

            print(f'Likelihood: forward={forward_ll}, backward={backward_ll}')

            # Update initial model
            idx = bounds_[:-1]
            in_ = X[idx, ...]
            out_ = np.exp(np.asarray(log_gamma_[idx, :]))
            self.initial_model.fit(in_, out_)

            # Update transition models
            in_ = X  # TODO: avoid connecting end of sequence i to start of sequence i+1
            out_ = np.exp(np.asarray(log_xi_))
            for i in range(self.n_states):
                self.transition_models[i].fit(in_, out_[:, i, :])

            # Update emission models
            in_ = X
            out_ = X
            sample_weights = np.exp(np.asarray(log_gamma_))
            for i in range(self.n_states):
                self.emission_models[i].fit(in_, out_, sample_weight=sample_weights[:, i])

    @property
    def n_states(self) -> int:
        assert len(self.emission_models) == len(self.transition_models)  # TODO
        return len(self.emission_models)
