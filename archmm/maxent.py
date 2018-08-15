# -*- coding: utf-8 -*-
# maxent.py
# author: Antoine Passemiers

import numpy as np

from abc import abstractmethod, ABCMeta

from archmm.ann.model import NeuralStackClassifier
from archmm.ann.layers import FullyConnected, Activation
from archmm.stats import np_data_t
from archmm.check_data import *
from archmm.utils import binarize_labels


class MaxEntClassifier(metaclass=ABCMeta):

    def __init__(self, n_features, n_states, **kwargs):
        self.n_features = n_features
        self.n_states = n_states

    @abstractmethod
    def _fit(self, X, y):
        pass
    
    def fit(self, X, y):
        n_unique = len(np.unique(y))
        if n_unique == 1:
            self.unique = y[0]
        else:
            self.unique = None
            self._fit(X, y)
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    
    @abstractmethod
    def _predict_proba(self, X):
        pass

    def predict_proba(self, X):
        if self.unique is not None: # TODO
            proba = np.zeros((len(X), self.n_states))
            proba[:, self.unique] = 1.
            return proba
        else:
            return self._predict_proba(X)


class LogisticRegression(MaxEntClassifier, NeuralStackClassifier):

    def __init__(self, n_features, n_states, **kwargs):
        NeuralStackClassifier.__init__(self, **kwargs)
        self.n_features = n_features
        self.n_states = n_states
        self.add(FullyConnected(n_features, n_states))
        self.add(Activation(func='softmax'))

    def _fit(self, X, y):
        if len(y.shape) == 1:
            y = binarize_labels(y, self.n_states)
        return NeuralStackClassifier.fit(self, X, y, max_n_iter=10)

    def _predict_proba(self, X):
        return self.eval(X)


class MEMM:
    """

    References:
        .. http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf
    """


    def __init__(self, n_states, base_maxent_class=LogisticRegression):
        self.base_maxent_class = base_maxent_class
        self.n_states = n_states
        self.start_maxent = None
        self.trans_maxents = list()

    def fit(self, X_s, Y_s):
        # TODO: handle missing values
        X_s, self.n_features = check_hmm_sequences_list(X_s)

        self.start_maxent = self.base_maxent_class(self.n_features, self.n_states)
        start_X = np.asarray([X[0] for X in X_s])
        start_y = np.asarray([y[0] for y in Y_s])
        self.start_maxent.fit(start_X, start_y)

        self.trans_maxents = list()
        for i in range(self.n_states):
            self.trans_maxents.append(
                self.base_maxent_class(self.n_features, self.n_states))
            
            X_i, y_i = list(), list()
            for k in range(len(Y_s)):
                indices = np.where(Y_s[k] == i)[0]
                if indices[-1] + 1 >= len(Y_s[k]):
                    indices = indices[:-1]
                X_i.append(X_s[k][indices])
                y_i.append(Y_s[k][indices+1])
            X_i = np.concatenate(X_i, axis=0)
            y_i = np.concatenate(y_i, axis=0)

            self.trans_maxents[i].fit(X_i, y_i)
    
    def score(self, X, y):
        log_likelihood = 0
        proba = self.start_maxent.predict_proba(X[0][np.newaxis, ...])[0]
        log_likelihood += np.log(proba[y[0]])
        for t in range(1, len(X)):
            state = y[t-1]
            proba = self.trans_maxents[state].predict_proba(X[t-1][np.newaxis, ...])[0]
            log_likelihood += np.log(proba[y[t]])
        return log_likelihood

    def predict(self, X):
        X, n_features = check_hmm_sequence(X)
        assert(self.n_features == n_features) # TODO: raise exception
        return self.viterbi(X)

    def viterbi(self, X):
        y_hat = np.empty(len(X), dtype=np.int)
        proba = self.start_maxent.predict_proba(X[0][np.newaxis, ...])[0]
        y_hat[0] = proba.argmax()

        for t in range(1, len(X)):
            state = y_hat[t-1]
            proba = self.trans_maxents[state].predict_proba(X[t-1][np.newaxis, ...])[0]
            y_hat[t] = proba.argmax()

        return y_hat