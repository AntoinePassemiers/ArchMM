# -*- coding: utf-8 -*-
# cython: profile=True

import pickle, numbers
import numpy as np
from numpy.random import randn, random, dirichlet

from libc.math cimport exp, log, M_PI

include "HMM.pyx"



def convertToMultiSpace(data):
    k = 0
    obs = dict()
    for el in data:
        if not isinstance(el, numbers.Number):
            if not el in obs.keys():
                obs[el] = k
                k += 1
            # TODO

class AdaptiveHMM:
    
    def __init__(self, n_states, architecture = ARCHITECTURE_LINEAR):
        self.hmm = BaseHMM(n_states, architecture = architecture)
        
    def getMu(self):
        return self.hmm.getMu()
        
    def fit(self, observations, **kwargs):
        # TODO : Process observations before passing them to self.hmm
        if len(observations.shape) == 1:
            obs = np.zeros((len(observations), 2), dtype = np.double)
            obs[:, 0] = observations[:]
            obs[:, 1] = np.random.rand(len(observations))
            observations = obs
        self.hmm.fit(observations, **kwargs)
        
    def score(self, *args, **kwargs):
        return self.hmm.score(*args, **kwargs)
        
    def randomSequence(self, *args, **kwargs):
        return self.hmm.randomSequence(*args, **kwargs)
        
    def pySave(self, *args, **kwargs):
        return self.hmm.pySave(*args, **kwargs)
        
    def pyLoad(self, *args, **kwargs):
        return self.hmm.pyLoad(*args, **kwargs)