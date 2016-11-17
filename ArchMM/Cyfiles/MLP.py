# -*- coding: utf-8 -*-

import numpy as np
import os, timeit

import theano
import theano.typed_list
from theano.tensor.shared_randomstreams import RandomStreams

from Theano_ops import *

theano.config.allow_gc = True
theano.config.scan.allow_gc = True
theano.config.scan.allow_output_prealloc = False
theano.config.exception_verbosity = 'high'
theano.config.profile = False
theano.config.profile_memory = False
theano.config.NanGuardMode.nan_is_error = False
theano.config.NanGuardMode.inf_is_error = True

RELEASE_MODE = False

SUBNETWORK_PI_STATE = 300
SUBNETWORK_STATE    = 301
SUBNETWORK_OUTPUT   = 302

ERGODIC_LAYER = 401
LINEAR_LAYER  = 402

DEBUG_MODE = theano.compile.MonitorMode(
    post_func = theano.compile.monitormode.detect_nan).excluding('local_elemwise_fusion', 'inplace')

class Layer:  
    def processOutput(self, X):
        linear_output = theano.tensor.dot(X, self.W) + self.b
        if self.activation is not None:
            output = self.activation(linear_output)
        else:
            output = linear_output
        return output
    def __getstate__(self):
        return (self.W, self.b)
    def __setstate__(self, state):
        self.W, self.b = state

class LogisticRegression(Layer):
    def __init__(self, input, n_in, n_out, rng = np.random.RandomState(1234)):
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
        
        self.p_y_given_x = theano.tensor.nnet.softmax(theano.tensor.dot(input, self.W) + self.b)
        self.y_pred = theano.tensor.argmax(self.p_y_given_x, axis = 1)
        self.params = [self.W, self.b]
        self.activation = theano.tensor.nnet.softmax
    def negative_log_likelihood(self, y):
        return -theano.tensor.mean(theano.tensor.log(self.p_y_given_x)[theano.tensor.arange(y.shape[0]), y])
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return theano.tensor.mean(theano.tensor.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
        
class HiddenLayer(Layer):
    def __init__(self, input, n_in, n_out, W = None, b = None,
                 activation = theano.tensor.nnet.nnet.sigmoid,
                 rng = np.random.RandomState(1234)):
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
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value = W_values, name = 'W', borrow = True)

        if b is None:
            b_values = np.asarray(np.random.rand(n_out), dtype = np.float32)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        
        self.W = W
        self.b = b

        lin_output = theano.tensor.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, n_in, n_hidden, n_out, rng = np.random.RandomState(1234), dropout_threshold = 0.5,
                 hidden_activation_function = "sigmoid"):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.input = self.x = theano.tensor.matrix(name = 'x', dtype = 'float32')
        self.rng = rng
        if hidden_activation_function == "tanh":
            self.activation = theano.tensor.tanh
        else:
            self.activation =  theano.tensor.nnet.nnet.sigmoid
        self.hiddenLayer = HiddenLayer(
            self.input, n_in, n_hidden, 
            rng = self.rng,
            activation = self.activation
        )
        self.logRegressionLayer = LogisticRegression(self.hiddenLayer.output, n_hidden, n_out)
        self.layers = [self.hiddenLayer, self.logRegressionLayer]
        self.L1 = (
            np.abs(self.hiddenLayer.W).sum()
            + np.abs(self.logRegressionLayer.W).sum()
        )
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input
        
        symbolic_X_j_k = theano.tensor.vector(name = "X", dtype = 'float32')
        next_layer_input = symbolic_X_j_k 
        for layer in self.layers:
            next_layer_input = layer.processOutput(next_layer_input)
        self.computeOutputFunction = theano.function(inputs = [symbolic_X_j_k], outputs = next_layer_input)
        
        self.dropout_threshold = dropout_threshold
        srng = RandomStreams(seed = 1234)
        rnd_number = srng.uniform(ndim = 0, low = 0.0, high = 1.0)
        self.dropout_or_nan_to_num = theano.tensor.switch(rnd_number > dropout_threshold, 0.00001, 0)
    
    def computeOutput(self, X):
        return self.computeOutputFunction(X)
    
    def processOutput(self, X):
        for layer in self.layers:
            X = layer.processOutput(X)
        return X
    
    def predict(self, test_X):
        X_values = np.asarray(test_X, dtype = np.float32)
        test_X = theano.shared(name = "X_test", borrow = True, value = X_values)
        return self.processOutput(test_X).eval()
    
    def __getstate__(self):
        state = list()
        for layer in self.layers:
            state.append(layer.__getstate__())
        return state
    
    def __setstate__(self, state):
        i = 0
        for layer in self.layers:
            layer.__setstate__(state[i])
            i += 1
    
class PiStateSubnetwork(MLP):
    def __init__(self, n_in, n_hidden, n_out, dropout_threshold = 0.5,
                 hidden_activation_function = "sigmoid", learning_rate = 0.01):
        MLP.__init__(self, n_in, n_hidden, n_out, 
                     hidden_activation_function = hidden_activation_function,
                     dropout_threshold = dropout_threshold)
        self.state_id = 0
        self.index = theano.tensor.lscalar('index')
        self.symbolic_gamma_j = theano.tensor.matrix(name = 'gamma_j', dtype = 'float32')
        self.train_set_x = theano.tensor.tensor3(name = 'x', dtype = 'float32')
        self.gamma = theano.tensor.tensor3(name = 'gamma', dtype = 'float32')
        self.learning_rate = theano.tensor.fscalar("learning_rate")
        self.s = theano.tensor.lscalar('s')

        self.cost = - theano.tensor.dot(self.gamma[self.s, :, 0], 
                        theano.tensor.log(self.processOutput(self.train_set_x[self.s, 0, :])[0]))
        
        # http://deeplearning.net/software/theano/sandbox/randomnumbers.html
        self.gparams = [theano.tensor.grad(self.cost, param) for param in self.params]
        self.updates = list()

        for param, gparam in zip(self.params, self.gparams):
            weight_update = param - self.learning_rate * gparam
            self.updates.append((param, theano.tensor.switch(theano.tensor.isnan(weight_update), 
                                self.dropout_or_nan_to_num, weight_update)))
        
        self.train_model = theano.function(
            inputs = [self.train_set_x, self.gamma, self.s, self.learning_rate],
            outputs = self.cost,
            updates = self.updates
        )
        
        if not RELEASE_MODE:
            debugfile = open("theano_pistatenetwork_graph.txt", "w")
            theano.printing.debugprint(self.cost, file = debugfile)
            debugfile.close()
    def train(self, train_set_x, gamma, n_epochs = 1, learning_rate = 0.01):
        epoch = 0
        N = len(train_set_x)
        while (epoch < n_epochs):
            epoch += 1
            avg_cost = 0
            for j in range(N):
                avg_cost += self.train_model(train_set_x, gamma, j, learning_rate)
            avg_cost /= N
        return avg_cost

class StateSubnetwork(MLP):
    def __init__(self, state_id, n_in, n_hidden, n_out, architecture = "ergodic", 
                 dropout_threshold = 0.5,
                 hidden_activation_function = "sigmoid", learning_rate = 0.01):
        MLP.__init__(self, n_in, n_hidden, n_out, 
                     hidden_activation_function = hidden_activation_function,
                     dropout_threshold = dropout_threshold)
        self.state_id = state_id
        self.index = theano.tensor.lscalar('index')
        self.t = theano.tensor.lscalar('t')
        self.s = theano.tensor.lscalar('s')
        self.symbolic_xi = theano.tensor.tensor4(name = 'xi', dtype = 'float32')
        self.symbolic_x = theano.tensor.tensor3(name = 'x', dtype = 'float32')
        self.learning_rate = theano.tensor.fscalar("learning_rate")
        
        phi = self.processOutput(self.symbolic_x[self.s, self.t, :])[0]
        self.cost = - (self.symbolic_xi[self.s, self.state_id, self.t, :] * theano.tensor.log(phi)).sum()
        
        self.gparams = [theano.tensor.grad(self.cost, param) for param in self.params]
        self.updates = list()
        for param, gparam in zip(self.params, self.gparams):
            weight_update = param - self.learning_rate * gparam
            self.updates.append((param, theano.tensor.switch(theano.tensor.isnan(weight_update), 
                                self.dropout_or_nan_to_num, weight_update)))
        
        if not RELEASE_MODE:
            debugfile = open("theano_statenetwork_graph.txt", "w")
            theano.printing.debugprint(self.cost, file = debugfile)
            
        self.train_model = theano.function(
            inputs = [self.symbolic_x, self.symbolic_xi, self.s, self.t, self.learning_rate],
            outputs = self.cost,
            updates = self.updates
        )
        
    def train(self, train_set_x, xi, is_mv, n_epochs = 1, learning_rate = 0.01):
        N = len(train_set_x)
        epoch = 0
        while (epoch < n_epochs):
            epoch += 1
            M = 0
            avg_cost = 0
            for sequence_id in range(N):
                for j in range(len(train_set_x[sequence_id])):
                    if not is_mv[sequence_id, j]:
                        cost = self.train_model(train_set_x, xi, sequence_id, j, learning_rate)
                        avg_cost += cost
                        M += 1
        return avg_cost / float(M)
                    
        
class OutputSubnetwork(MLP):
    def __init__(self, state_id, n_in, n_hidden, n_out, is_classifier = True,
                 dropout_threshold = 0.5, 
                 hidden_activation_function = "sigmoid", learning_rate = 0.01):
        MLP.__init__(self, n_in, n_hidden, n_out, 
                     hidden_activation_function = hidden_activation_function,
                     dropout_threshold = dropout_threshold)
        self.state_id = state_id
        self.is_classifier = is_classifier
        self.memory = None
        self.index = theano.tensor.lscalar('index')
        self.t = theano.tensor.lscalar('t')
        self.s = theano.tensor.lscalar('s')
        self.symbolic_memory = theano.tensor.tensor3(name = 'memory', dtype = 'float32')
        self.symbolic_x = theano.tensor.tensor3(name = 'x_j', dtype = 'float32')
        self.train_set_x = theano.tensor.matrix(name = 'x', dtype = 'float32')
        self.learning_rate = theano.tensor.fscalar("learning_rate")
        self.symbolic_target = theano.tensor.imatrix(name = "target")
        
        eta = self.processOutput(self.symbolic_x[self.s, self.t, :])[0]
        self.cost = - (self.symbolic_memory[self.s, self.t, self.state_id] * \
                       theano.tensor.log(eta[self.symbolic_target[self.s, self.t]])).sum()
        
        self.gparams = [theano.tensor.grad(self.cost, param) for param in self.params]
        self.updates = list()
        for param, gparam in zip(self.params, self.gparams):
            weight_update = param - self.learning_rate * gparam
            self.updates.append((param, theano.tensor.switch(theano.tensor.isnan(weight_update), 
                                self.dropout_or_nan_to_num, weight_update)))
        
        self.train_model = theano.function(
            inputs = [self.symbolic_x, self.symbolic_target, self.symbolic_memory, self.s, self.t, self.learning_rate],
            outputs = self.cost,
            updates = self.updates
        )
        
        if not RELEASE_MODE:
            debugfile = open("theano_outputnetwork_graph.txt", "w")
            theano.printing.debugprint(self.cost, file = debugfile)
        
        
    def train(self, train_set_x, target_set, memory_array, is_mv, n_epochs = 1, learning_rate = 0.05):
        N = len(train_set_x)
        assert(N == len(target_set))
        epoch = 0
        while (epoch < n_epochs):
            epoch += 1
            M = 0
            avg_cost = 0
            for sequence_id in range(N):
                for j in range(len(train_set_x[sequence_id])):
                    if not is_mv[sequence_id, j]:
                        cost = self.train_model(train_set_x, target_set,
                                memory_array, sequence_id, j, learning_rate)
                        avg_cost += cost
                        M += 1
        return avg_cost / float(M)

def new3DVLMArray(P, T, ndim = 0, ndim_2 = 0, dtype = np.double):
    if isinstance(ndim, int):
        return np.empty((P, T.max()), dtype = dtype)
    elif ndim_2 == 0:
        if type(ndim) == int:
            return np.empty((P, T.max(), ndim), dtype = dtype)
        else:
            return np.empty((P, T, ndim.max()), dtype = dtype)
    else:
        return np.empty((P, T, ndim.max(), ndim_2), dtype = dtype)

def typedListToPaddedTensor(typed_list, T, is_3D = True, dtype = np.float32):
    assert(len(typed_list) > 0)
    if isinstance(typed_list, list):
        P = len(T)
        n_max = T.max()
        if is_3D:
            n_dim = typed_list[0].shape[1]
            new_tensor = np.zeros((P, n_max, n_dim), dtype = dtype)
            for p in range(P):
                new_tensor[p, :T[p], :] = typed_list[p][:, :]
        else:
            new_tensor = np.zeros((P, n_max), dtype = dtype)
            for p in range(P):
                new_tensor[p, :T[p]] = typed_list[p][:]
        return new_tensor
    else:
        return typed_list
    