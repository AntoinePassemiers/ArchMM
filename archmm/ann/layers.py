# -*- coding: utf-8 -*-
# layers.py
# author: Antoine Passemiers

import numpy as np
from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
        
    @abstractmethod
    def forward(self, X):
        pass
    
    @abstractmethod
    def backward(self, signal):
        pass


class FullyConnected(Layer):

    def __init__(self, n_in, n_out, with_bias=True, dtype=np.double):
        self.n_in = n_in
        self.n_out = n_out
        self.with_bias = with_bias
        self.dtype = dtype
        self.weights = self.biases = None
        self.current_input = None
        self.__initialize()
    
    def __initialize(self):
        # Weights: Glorot Uniform initialization
        limit = np.sqrt(2. / (self.n_in + self.n_out))
        self.weights = np.random.uniform(
            -limit, limit, size=(self.n_in, self.n_out)).astype(self.dtype)
        # Biases : Zero initialization
        if self.with_bias:
            self.biases = np.zeros((1, self.n_out), dtype=self.dtype)
    
    def forward(self, X):
        self.current_input = X
        return np.dot(X, self.weights) + self.biases
    
    def backward(self, signal):
        gradient_weights = np.dot(self.current_input.T, signal)
        gradient_biases = np.sum(signal, axis=0, keepdims=True)
        # TODO


class Activation(Layer):

    def __init__(self, func='sigmoid'):
        self.func = func
    
    def forward(self, X):
        if self.func == 'sigmoid':
            out = 1. / (1. + np.exp(-X))
        elif self.func == 'tanh':
            out = np.tanh(X)
        elif self.func == 'relu':
            out = np.maximum(X, 0)
        elif self.func == 'softmax':
            e = np.exp(X)
            out = e / np.sum(e, axis=1, keepdims=True)
        else:
            raise NotImplementedError()
        return out

    def backward(self, signal):
        X = self.current_output
        if self.func == 'sigmoid':
            grad_X = X * (1. - X)
        elif self.func == 'tanh':
            grad_X = 1. - X ** 2
        elif self.func == 'relu':
            grad_X = (self.current_input >= 0)
        elif self.function == 'softmax':
            grad_X = X * (1. - X)
        else:
            raise NotImplementedError()
        return grad_X * signal