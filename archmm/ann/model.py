# -*- coding: utf-8 -*-
# model.py
# author: Antoine Passemiers

from archmm.ann.layers import *

import numpy as np
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


class CrossEntropy:

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def eval(self, y, y_hat):
        indices = np.argmax(y, axis=1).astype(np.int)
        predictions = y_hat[np.arange(len(y_hat)), indices]
        log_predictions = np.log(np.maximum(predictions, self.epsilon))
        return -np.mean(log_predictions)

    def grad(self, y, y_hat):
        return y_hat - y


class NeuralStackClassifier:

    def __init__(self, optimizer=Optimizer(1)):
        self.layers = list()
        self.optimizer = optimizer
        self.cost = CrossEntropy()

    def add(self, layer):
        self.layers.append(layer)

    def eval(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def fit(self, X, y, max_n_iter=100):
        print()
        for k in range(max_n_iter):
            batch_X, batch_y = X, y # TODO: batching
            y_hat = self.eval(X)

            print(self.cost.eval(batch_y, y_hat))

            signal = self.cost.grad(batch_y, y_hat)

            for layer in reversed(self.layers):
                signal, gradient = layer.backward(signal)
                if gradient is not None:
                    self.optimizer.add_gradient_fragments(layer, gradient)
            self.optimizer.update()
