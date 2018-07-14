# -*- coding: utf-8 -*-
# subnetworks.py
# author: Antoine Passemiers

from archmm.ann.layers import *

import numpy as np
from abc import ABCMeta, abstractmethod


class MultiLayerPerceptron:

    def __init__(self, n_in, n_hidden, n_out,
                 hidden_activation='sigmoid', is_classifier=True):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.hidden_activation = hidden_activation
        self.is_classifier = is_classifier

        self.layers = list()
        self.layers.append(FullyConnected(n_in, n_hidden))
        self.layers.append(Activation(func=self.hidden_activation))
        self.layers.append(FullyConnected(n_hidden, n_out))
        if self.is_classifier:
            self.layers.append(Activation(func='softmax'))
    
    def eval(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X


class Subnetwork(metaclass=ABCMeta):
    
    @abstractmethod
    def eval(self, X):
        pass
    

class StartSubnetwork(Subnetwork):

    def __init__(self):
        pass


class TransitionSubnetwork(Subnetwork):

    def __init__(self):
        pass


class EmissionSubnetwork(Subnetwork):

    def __init__(self):
        pass


class StartMLP(StartSubnetwork, MultiLayerPerceptron):

    def __init__(self, *args, **kwargs):
        MultiLayerPerceptron.__init__(self, *args, **kwargs)
    
    def eval(self, X):
        return MultiLayerPerceptron.eval(self, X)


class TransitionMLP(TransitionSubnetwork, MultiLayerPerceptron):

    def __init__(self, *args, **kwargs):
        MultiLayerPerceptron.__init__(self, *args, **kwargs)

    def eval(self, X):
        return MultiLayerPerceptron.eval(self, X)


class EmissionMLP(EmissionSubnetwork, MultiLayerPerceptron):

    def __init__(self, *args, **kwargs):
        MultiLayerPerceptron.__init__(self, *args, **kwargs)

    def eval(self, X):
        return MultiLayerPerceptron.eval(self, X)