# -*- coding: utf-8 -*-
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=True

import collections, numbers, pickle, json
import numpy as np

from archmm.hmm import *

def id_generator(dict):
    for k, v in dict.items():
        if k != "description":
            yield v
        elif isinstance(v, dict):
            for id_val in id_generator(v):
                yield id_val
        elif isinstance(v, list):
            pass # TODO

class IOConfig:
    def __init__(self, config_filepath = None):
        try:
            if not config_filepath:
                config_filepath = "default_ioconfig.json"
            config_file = open(config_filepath, 'r')
            data = json.load(config_file)
            for value in id_generator(data):
                print(value)
            close(config_file)
        except:
            print("Warning : default ioconfig file cound not be found")

class HMM:
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
                 missing_value = DEFAULT_MISSING_VALUE, has_io = False):
        l_architecture = architecture.lower()
        found = False
        for name in HMM.arch_names.keys():
            if l_architecture in name:
                arch = HMM.arch_names[name]
                found = True
        if not found:
            arch = HMM.arch_names["ergodic"]
            print("Warning : architecture %s not found" % l_architecture)
            print("An ergodic structure will be used instead.")
        
        self.hmm = BaseHMM(n_states, architecture = arch, missing_value = missing_value) 
        self.stdvs = self.mu = None
        self.standardize = standardize
        self.has_io = has_io
        self.mu = self.sigma = None

    def __getitem__(self, attr):
        if attr in self.hmm.__names__:
            return self.hmm.__getitem__(attr)
        else:
            return self.__dict__[attr]

    def __setitem__(self, attr, value):
        if attr in self.hmm.__names__:
            self.hmm.__setitem__(attr, value)
        else:
            self.__dict__[attr] = value
        
    def getMu(self):
        return self.hmm.getMu()
    
    def getA(self):
        return self.hmm.getA()
        
    def fit(self, observations, **kwargs):
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
            return self.hmm.fit(observations, self.mu, self.sigma, **kwargs)
        else:

            K = kwargs.keys()
            for arg in ("targets", "is_classifier"):
                if not arg in K:
                    print("Error. Parameter [[%s]] must be provided for IO-HMM." % arg)
                    exit(EXIT_FAILURE)
            self.sigma = np.cov(observations[0].T)
            self.stdvs = np.sqrt(np.diag(self.sigma))
            self.mu = np.mean(observations[0], axis = 0)
            if 0 and self.standardize:
                for i in range(len(observations)):
                    observations[i] = (observations[i] - self.mu) / self.stdvs
            return self.hmm.fitIO(observations, mu = self.mu, sigma = self.sigma, **kwargs)
        
    def score(self, observations, mode = "aicc"):
        l_mode = mode.lower()
        found = False
        for name in HMM.crit_names.keys():
            if l_mode in name:
                mode = HMM.crit_names[name]
                found = True
        if not found:
            mode = HMM.crit_names["likelihood"]
            print("Warning : mode %s not found" % l_mode)
            print("The likelihood will be returned instead.")
            
        if self.standardize:
            observations = (observations - self.mu) / self.stdvs
        return self.hmm.score(observations, mode = mode)
    
    def predictIO(self, *args, **kwargs):
        return self.hmm.predictIO(*args, **kwargs)
    
    def randomSequence(self, *args, **kwargs):
        return self.hmm.randomSequence(*args, **kwargs)
        
    def pySave(self, filename):
        attributes = {
            "MU" : self.mu, 
            "SIGMA" : self.sigma,
            "stdvs" : self.stdvs,
            "standardize" : self.standardize,
            "has_io" : self.has_io
        }
        if not self.has_io:
            self.hmm.pySave(filename)
        else:
            self.hmm.saveIO(filename)
        try:
            pickle.dump(attributes, open(filename + "_adapt", "wb"))
        except MemoryError:
            pickle.dump(dict(), open(filename + "_adapt", "wb"))

    def pyLoad(self, filename):
        try:
            attributes = pickle.load(open(filename + "_adapt", "rb"))
            self.mu = attributes["MU"]
            self.sigma = attributes["SIGMA"]
            self.stdvs = attributes["stdvs"]
            self.standardize = attributes["standardize"]
            self.has_io = attributes["has_io"]
        except:
            pass
        if not self.has_io:
            self.hmm.pyLoad(filename)
        else:
            self.hmm.loadIO(filename)