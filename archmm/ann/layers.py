# -*- coding: utf-8 -*-

import numpy as np
import os, timeit

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams

from archmm.utils import requiresTheano


theano.config.floatX = 'float32'
theano.config.allow_gc = True
theano.config.scan.allow_gc = True
theano.config.scan.allow_output_prealloc = False
theano.config.exception_verbosity = 'high'
theano.config.profile = False
theano.config.profile_memory = False
theano.config.NanGuardMode.nan_is_error = False
theano.config.NanGuardMode.inf_is_error = True

class Layer:
    @requiresTheano(True)
    def __init__(self):
        pass
    def processOutput(self, X):
        linear_output = T.dot(X, self.W) + self.b
        if self.activation is not None:
            output = self.activation(linear_output)
        else:
            output = linear_output
        return output
    def __getstate__(self):
        return (self.W.get_value(), self.b.get_value())
    def __setstate__(self, state):
        self.W.set_value(state[0])
        self.b.set_value(state[1])
        
        
class LogisticRegression(Layer):
    @requiresTheano(True)
    def __init__(self, input, n_in, n_out, rng = np.random.RandomState(1234)):
        self.is_conv = False
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        W_values = np.asarray(
            rng.uniform(
                low  = -np.sqrt(6. / (n_in + n_out)),
                high = np.sqrt(6. / (n_in + n_out)),
                size = (n_in, n_out)
            ),
            dtype = np.float32)
        W_values *= 4
        self.W = theano.shared(value = W_values, name = 'W', borrow = True)
        b_values = np.asarray(np.random.rand(n_out), dtype = np.float32)
        self.b = theano.shared(value = b_values, name = 'b', borrow = True)
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.output = self.p_y_given_x
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        self.params = [self.W, self.b]
        self.activation = T.nnet.softmax
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
        
class HiddenLayer(Layer):
    @requiresTheano(True)
    def __init__(self, input, n_in, n_out, W = None, b = None,
                 activation = T.nnet.nnet.sigmoid,
                 rng = np.random.RandomState(1234)):
        self.n_in, self.n_out = n_in, n_out
        self.is_conv = False
        self.input = input
        self.activation = activation
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6. / (n_in + n_out)),
                    high = np.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype = np.float32
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value = W_values, name = 'W', borrow = True)

        if b is None:
            b_values = np.asarray(np.random.rand(n_out), dtype = np.float32)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]
        
    
