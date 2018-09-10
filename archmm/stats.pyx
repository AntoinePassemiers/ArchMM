# -*- coding: utf-8 -*-
# stats.pyx
# author: Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

import scipy.stats

from archmm.check_data import is_iterable


np_data_t = np.double


def mat_stable_inv(mat):
    try:
        mat = np.nan_to_num(mat) # TODO
        cholesky = scipy.linalg.cholesky(
            mat, lower=True, check_finite=True)
    except scipy.linalg.LinAlgError:
        # TODO: warning (singular matrix) 
        # -> potentially highly correlated features
        mcv = 1.e-7
        is_not_spd = True
        while is_not_spd:
            try:
                cholesky = scipy.linalg.cholesky(
                    mat + mcv * np.eye(mat.shape[0]),
                    lower=True, check_finite=True)
                is_not_spd = False
            except scipy.linalg.LinAlgError:
                mcv *= 10
    log_det = 2 * np.sum(np.log(np.diagonal(cholesky)))
    return cholesky, log_det


def gaussian_log_proba(X, mu, sigma):
    n_features = X.shape[1]
    cholesky, log_det = mat_stable_inv(sigma)
    mahalanobis = scipy.linalg.solve_triangular(
        cholesky, (np.asarray(X) - np.asarray(mu)).T, lower=True).T
    lnf = -0.5 * (np.sum(mahalanobis ** 2, axis=1) + \
        n_features * np.log(2 * np.pi) + log_det)
    return lnf.astype(np_data_t)


class MCMC:

    def __init__ (self, pdf, start, proposal=None):
        self.pdf = pdf

        if proposal is not None:
            self.proposal = proposal
        else:
            self.proposal = scipy.stats.norm()
        
        self.current_point = start
        if is_iterable(self.current_point):
            self.n_features = len(self.current_point)
        else:
            self.n_features = 1
    
    def acceptance_rule(self, new_point):
        eta = np.random.rand()
        if self.pdf(new_point) / self.pdf(self.current_point) > eta:
            return new_point
        else:
            return self.current_point
        
    def sample(self, n_samples):
        shape = (n_samples, self.n_features) if self.n_features > 1 else (n_samples,)
        samples = np.empty(shape, dtype=self.current_point.dtype)

        for k in range(n_samples):
            next_point = self.current_point + self.proposal.rvs()
            self.current_point = samples[k] = self.acceptance_rule(next_point)
        return samples