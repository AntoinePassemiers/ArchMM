# -*- coding: utf-8 -*-
# layers.py
# author: Antoine Passemiers

__all__ = ['Layer', 'FullyConnected', 'Activation']

import numpy as np
from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    """ Base class for neural network layers.

    Each layer must implement both forward pass
    and backward pass. These methods are called
    automatically by neural networks in order
    to evaluate the outputs or to apply a backward pass.
    """

    def __init__(self):
        self.back_propagate = True
        
    def deactivate_backpropagation(self):
        self.back_propagate = False

    @abstractmethod
    def forward(self, X):
        pass
    
    @abstractmethod
    def backward(self, signal):
        pass


class FullyConnected(Layer):
    """ Fully-connected (dense) layer where each input is connected
    to each neuron with an associated weight. Weights are organised
    in an array of shape (n_in, n_out), where n_in is the number
    of inputs to the layer and n_out is the number of neurons.

    Args:
        n_in (int):
            Number of inputs to the layer
        n_out (int):
            Number of neurons
        with_bias (bool):
            Whether to add biases to the outputs
        dtype (type):
            Data type of the weights and biases
    
    Attributes:
        weights (:obj:`np.ndarray`):
            Connection weights
        biases (:obj:`np.ndarray`):
            Biases to add to the outputs (if with_bias is True)
        current_input (:obj:`np.ndarray`):
            Last input tensor obtained during a forward pass.
            This is used to compute the gradient of the loss
            function w.r.t. layer parameters during backward pass.
    """

    def __init__(self, n_in, n_out, with_bias=True, dtype=np.double):
        Layer.__init__(self)
        self.n_in = n_in
        self.n_out = n_out
        self.with_bias = with_bias
        self.dtype = dtype
        self.weights = self.biases = None
        self.current_input = None
        self.initialize()
    
    def initialize(self):
        """ Initializes the connection weights using Glorot Uniform Initializer.
        If with_bias is True, initializes the biases to zeros.
        """

        # Weights: Glorot Uniform initialization
        limit = np.sqrt(2. / (self.n_in + self.n_out))
        self.weights = np.random.uniform(
            -limit, limit, size=(self.n_in, self.n_out)).astype(self.dtype)
        # Biases : Zero initialization
        if self.with_bias:
            self.biases = np.zeros((1, self.n_out), dtype=self.dtype)
    
    def forward(self, X):
        r""" Apply a forward pass on input X.

        Because the layer is fully-connected, outputs can be expressed as
        linear combinations of all inputs or as matrix multiplication:

        .. math::
            O_{i, j} = \sum\limits_{k=1}^m X_{i, k} W_{k, j} + b_{j}

        where O is the output matrix, X the input matrix, W the connection
        weights and b the bias vector. b is added to the outputs when
        with_bias is set to True.

        Args:
            X (:obj:`np.ndarray`):
                Array of shape (n_samples, n_features)
        
        Returns:
            :obj:`np.ndarray`:
                Layer output
        """
        self.current_input = X
        return np.dot(X, self.weights) + self.biases
    
    def backward(self, signal):
        gradient_weights = np.dot(self.current_input.T, signal)
        gradient_biases = np.sum(signal, axis=0, keepdims=True)
        # TODO
        if self.with_bias:
            gradients = (gradient_weights, gradient_biases)
        else:
            gradients = gradient_weights
        if self.back_propagate:
            signal = np.dot(signal, self.weights.T)
        else:
            signal = None
        return (signal, gradients)

    def update_parameters(self, delta_fragments):
        self.weights -= delta_fragments[0]
        if self.with_bias:
            self.biases -= delta_fragments[1]


class Activation(Layer):

    def __init__(self, func='sigmoid'):
        Layer.__init__(self)
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
        self.current_output = out
        return out

    def backward(self, signal):
        if self.back_propagate:
            X = self.current_output
            if self.func == 'sigmoid':
                grad_X = X * (1. - X)
            elif self.func == 'tanh':
                grad_X = 1. - X ** 2
            elif self.func == 'relu':
                grad_X = (self.current_input >= 0)
            elif self.func == 'softmax':
                grad_X = X * (1. - X)
            else:
                raise NotImplementedError()
            out = grad_X * signal
            return (out, None)
        else:
            return (None, None)