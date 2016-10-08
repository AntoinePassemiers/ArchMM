# -*- coding: utf-8 -*-
# cython: profile=True

import collections, numbers, pickle
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
    crit_names["neg likelihood"] = CRITERION_NEG_LIKELIHOOD
    dist_names = collections.defaultdict()
    dist_names["gaussian"] = DISTRIBUTION_GAUSSIAN
    dist_names["multinomial"] = DISTRIBUTION_MULTINOMIAL
    
    def __init__(self, n_states, architecture = "linear", standardize = False,
                 missing_value = DEFAULT_MISSING_VALUE, has_io = False,
                 use_implementation = "linear"):
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
        
        use_implementation = USE_LIN_IMPLEMENTATION if use_implementation == "linear" else USE_LOG_IMPLEMENTATION
        self.hmm = BaseHMM(n_states, architecture = arch, missing_value = missing_value, 
                           use_implementation = use_implementation) 
        self.stdvs = self.mu = None
        self.standardize = standardize
        self.has_io = has_io
        
    def getMu(self):
        return self.hmm.getMu()
    
    def getA(self):
        return self.hmm.getA()
        
    def fit(self, observations, **kwargs):
        assert(not np.any(np.isnan(observations)))
        assert(not np.any(np.isinf(observations)))
        if not self.has_io:
            self.sigma = np.cov(observations.T)
            self.stdvs = np.sqrt(np.diag(self.sigma))
            self.mu = np.mean(observations, axis = 0)
            if self.standardize:
                observations = (observations - self.mu) / self.stdvs
            if len(observations.shape) == 1:
                obs = np.zeros((len(observations), 2), dtype = np.double)
                obs[:, 0] = observations[:]
                obs[:, 1] = np.random.rand(len(observations))
                observations = obs
            self.hmm.fit(observations, self.mu, self.sigma, **kwargs)
        else:

            K = kwargs.keys()
            for arg in ("targets", "n_classes"):
                if not arg in K:
                    print("Error. Parameter [[%s]] must be provided for IO-HMM." % arg)
                    return
            self.sigma = np.cov(observations[0].T)
            self.stdvs = np.sqrt(np.diag(self.sigma))
            self.mu = np.mean(observations[0], axis = 0)
            if self.standardize:
                for i in range(len(observations)):
                    observations[i] = (observations[i] - self.mu) / self.stdvs
            self.hmm.fitIO(observations, mu = self.mu, sigma = self.sigma, **kwargs)
        
    def score(self, observations, mode = "aicc"):
        l_mode = mode.lower()
        found = False
        for name in AdaptiveHMM.crit_names.keys():
            if l_mode in name:
                mode = AdaptiveHMM.crit_names[name]
                found = True
        if not found:
            mode = AdaptiveHMM.crit_names["likelihood"]
            print("Warning : mode %s not found" % l_mode)
            print("The likelihood will be returned instead.")
            
        if self.standardize:
            observations = (observations - self.mu) / self.stdvs
        return self.hmm.score(observations, mode = mode)
        
    def randomSequence(self, *args, **kwargs):
        return self.hmm.randomSequence(*args, **kwargs)
        
    def pySave(self, filename): # Save the whole object
        attributes = {
            "MU" : self.mu, 
            "SIGMA" : self.sigma,
            "stdvs" : self.stdvs,
            "standardize" : self.standardize,
            "has_io" : self.has_io
        }
        pickle.dump(attributes, open(filename + "_adapt", "wb"))
        self.hmm.pySave(<char*>filename)
        
    def pyLoad(self, filename): # Load the whole object
        attributes = pickle.load(open(filename + "_adapt", "rb"))
        self.mu = attributes["MU"]
        self.sigma = attributes["SIGMA"]
        self.stdvs = attributes["stdvs"]
        self.standardize = attributes["standardize"]
        self.has_io = attributes["has_io"]
        self.hmm.pyLoad(<char*>filename)
    
    def cSave(self, filepath):
        self.hmm.cSave(<char*>filepath)
        
    def cLoad(self, filepath):
        pass