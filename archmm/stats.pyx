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


class MCMC:

    def __init__ (self, pdf, start, proposal=None, dtype=np.double):
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
        self.dtype = dtype
    
    def acceptance_rule(self, new_point):
        eta = np.random.rand()
        if self.pdf(new_point) / self.pdf(self.current_point) > eta:
            return new_point
        else:
            return self.current_point
        
    def sample(self, n_samples):
        shape = (n_samples, self.n_features) if self.n_features > 1 else (n_samples,)
        samples = np.empty(shape, dtype=self.dtype)

        for k in range(n_samples):
            next_point = self.current_point + self.proposal.rvs()
            self.current_point = samples[k] = self.acceptance_rule(next_point)
        return samples