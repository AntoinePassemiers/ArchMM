# -*- coding: utf-8 -*-
# subnetworks.py
# author: Antoine Passemiers

__all__ = [
    'MultiLayerPerceptron',
    'Subnetwork',
    'StartSubnetwork',
    'TransitionSubnetwork',
    'EmissionSubnetwork',
    'StartMLP',
    'TransitionMLP',
    'EmissionMLP']

from archmm.ann.layers import *

import numpy as np
from abc import ABCMeta, abstractmethod
from operator import mul
from functools import reduce




class Optimizer:

    def __init__(self, learning_rate=.1, momentum=.2):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.previous_grad = None
        self.gradient_fragments = list()

    def update(self):
        gradient = list()
        for _, _, fragments in self.gradient_fragments:
            for fragment in fragments:
                gradient.append(fragment.flatten(order='C'))
        gradient = np.concatenate(gradient)

        delta = self.learning_rate * gradient
        if self.momentum > 0:
            if self.previous_grad is not None:
                delta += self.momentum * self.previous_grad
            self.previous_grad = delta

        self.update_layers(delta)
        self.gradient_fragments = list()

    def update_layers(self, delta):
        cursor = 0
        for src_layer, layer_param_shapes, _ in self.gradient_fragments:
            layer_fragments = list()
            for fragment_shape in layer_param_shapes:
                n_elements = reduce(mul, fragment_shape)
                fragment = delta[cursor:cursor+n_elements]
                layer_fragments.append(fragment.reshape(fragment_shape, order='C'))
                cursor += n_elements
            src_layer.update_parameters(tuple(layer_fragments))

    def add_gradient_fragments(self, src_layer, fragments):
        if not isinstance(fragments, (tuple, list)):
            fragments = [fragments]
        layer_param_shapes = list()
        for fragment in fragments:
            layer_param_shapes.append(fragment.shape)
        self.gradient_fragments.append((src_layer, layer_param_shapes, fragments))



class MultiLayerPerceptron:
    """ Multi-layer perceptron (MLP).

    Args:
        n_in (int):
            Number of inputs
        n_hidden (int):
            Number of hidden neurons
        n_out (int):
            Number of output neurons
        hidden_activation (str):
            Name of the activation function to be
            used by the hidden layer
        is_classifier (bool):
            Whether the MLP is a classifier. If classifier,
            a softmax activation function is applied on
            the output values.
    
    Attributes:
        layers (list):
            A list of :obj:`Layer` instances. The MLP behaves
            like a stack, by propagating the input from the first
            layer of the list to the last one. During backward pass,
            the signal is backpropagated from last layer of the
            list to the first one.
    """

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
        
        self.optimizer = Optimizer()
    
    def eval(self, X):
        """ Given inputs X, evaluate the output of the MLP.

        Args:
            X (:obj:`np.ndarray`):
                Array of shape (n_samples, n_features)
        
        Returns:
            :obj:`np.ndarray`:
                Array of shape (n_samples, output_dim)
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backpropagation(self, y_hat):
        signal = y_hat
        for layer in reversed(self.layers):
            signal, gradient = layer.backward(signal)
            if gradient is not None:
                self.optimizer.add_gradient_fragments(layer, gradient)
        self.optimizer.update()
    

class Subnetwork(metaclass=ABCMeta):
    
    @abstractmethod
    def eval(self, X):
        pass
    

class StartSubnetwork(Subnetwork):

    def __init__(self):
        pass
    
    @abstractmethod
    def train(self, gamma_c):
        pass


class TransitionSubnetwork(Subnetwork):

    def __init__(self):
        pass

    @abstractmethod
    def train(self, X_c, xi_j_c):
        pass


class EmissionSubnetwork(Subnetwork):

    def __init__(self):
        pass
    
    @abstractmethod
    def train(self, X_c, memory_j_c):
        pass


class StartMLP(StartSubnetwork, MultiLayerPerceptron):

    def __init__(self, *args, **kwargs):
        kwargs['is_classifier'] = True
        MultiLayerPerceptron.__init__(self, *args, **kwargs)
    
    def eval(self, X):
        return MultiLayerPerceptron.eval(self, X)
    
    def train(self, gamma_c):
        pass


class TransitionMLP(TransitionSubnetwork, MultiLayerPerceptron):

    def __init__(self, *args, **kwargs):
        kwargs['is_classifier'] = True
        MultiLayerPerceptron.__init__(self, *args, **kwargs)

    def eval(self, X):
        return MultiLayerPerceptron.eval(self, X)
    
    def train(self, X_c, xi_j_c):
        assert(X_c.shape[0] == xi_j_c.shape[0])
        indices = np.arange(len(X_c))
        np.random.shuffle(indices)
        indices = indices[:32] # TODO

        X_batch = X_c[indices]
        phi_batch = MultiLayerPerceptron.eval(self, X_batch)
        h_batch = xi_j_c[indices]

        MultiLayerPerceptron.backpropagation(self, - h_batch / phi_batch)


class EmissionMLP(EmissionSubnetwork, MultiLayerPerceptron):

    def __init__(self, *args, **kwargs):
        MultiLayerPerceptron.__init__(self, *args, **kwargs)

    def eval(self, X):
        return MultiLayerPerceptron.eval(self, X)
    
    def train(self, X_c, memory_j_c):
        assert(X_c.shape[0] == memory_j_c.shape[0])
        indices = np.arange(len(X_c))
        np.random.shuffle(indices)
        indices = indices[:32] # TODO

        X_batch = X_c[indices]
        eta_batch = MultiLayerPerceptron.eval(self, X_batch)
        memory_batch = memory_j_c[indices]

        MultiLayerPerceptron.backpropagation(self, (eta_batch.T * memory_batch).T)
