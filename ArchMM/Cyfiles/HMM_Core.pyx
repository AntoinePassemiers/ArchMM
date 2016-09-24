# -*- coding: utf-8 -*-
# cython: profile=True

import collections, numbers
import numpy as np

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

    arch_names = collections.defaultdict()
    arch_names["linear"] = ARCHITECTURE_LINEAR
    arch_names["right"] = ARCHITECTURE_LEFT_TO_RIGHT
    arch_names["ergodic"] = ARCHITECTURE_ERGODIC
    arch_names["cyclic"] = ARCHITECTURE_CYCLIC
    arch_names["bakis"] = ARCHITECTURE_BAKIS
    crit_names = collections.defaultdict()
    crit_names["aic"] = CRITERION_AIC
    crit_names["aicc"] = CRITERION_AICC
    crit_names["bic"] = CRITERION_BIC
    crit_names["likelihood"] = CRITERION_LIKELIHOOD
    dist_names = collections.defaultdict()
    dist_names["gaussian"] = DISTRIBUTION_GAUSSIAN
    dist_names["multinomial"] = DISTRIBUTION_MULTINOMIAL
    
    def __init__(self, n_states, architecture = "linear"):
        l_architecture = architecture.lower()
        found = False
        for name in AdaptiveHMM.arch_names.keys():
            if l_architecture in name:
                arch = AdaptiveHMM.arch_names[name]
                found = True
        if not found:
            arch = AdaptiveHMM.arch_names["ergodic"]
            print("Warning : architecture %s not found" % l_architecture)
            print("An ergodic structure will be used instead.")
        
        self.hmm = BaseHMM(n_states, architecture = arch)
        
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