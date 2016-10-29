# -*- coding: utf-8 -*-

import numpy as np
import os, timeit

from Utils import *
from numpy import dtype

os.environ["THEANO_FLAGS"] = "floatX=float32,exception_verbosity=high" # TODO : doesn't work

import theano
import theano.typed_list

SUBNETWORK_PI_STATE = 300
SUBNETWORK_STATE    = 301
SUBNETWORK_OUTPUT   = 302

ERGODIC_LAYER = 401
LINEAR_LAYER  = 402

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
        W_values = np.asarray(
            rng.uniform(
                low  = -np.sqrt(6. / (n_in + n_out)),
                high = np.sqrt(6. / (n_in + n_out)),
                size = (n_in, n_out)
            ),
            dtype=theano.config.floatX)
        W_values *= 4
        self.W = theano.shared(value = W_values, name='W', borrow=True)
        b_values = np.asarray(np.random.rand(n_out), dtype=theano.config.floatX)
        self.b = theano.shared(value = b_values, name='b', borrow=True)
        
        self.p_y_given_x = theano.tensor.nnet.softmax(theano.tensor.dot(input, self.W) + self.b)
        self.y_pred = theano.tensor.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input
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
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=theano.tensor.nnet.nnet.sigmoid):
        self.input = input
        self.activation = activation
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.asarray(np.random.rand(n_out), dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = theano.tensor.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, n_in, n_hidden, n_out, rng = np.random.RandomState(1234)):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.input = self.x = theano.tensor.matrix('x')
        self.rng = rng
        self.hiddenLayer = HiddenLayer(
            rng=self.rng,
            input=self.input,
            n_in=n_in,
            n_out=n_hidden,
            activation=theano.tensor.tanh
        )
        self.logRegressionLayer = LogisticRegression(
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_out
        )
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
    
    def processOutput(self, X):
        for layer in self.layers:
            X = layer.processOutput(X)
        return X
    
    def predict(self, test_X):
        X_values = np.asarray(test_X, dtype = theano.config.floatX)
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
    def __init__(self, n_in, n_hidden, n_out, learning_rate = 0.01):
        MLP.__init__(self, n_in, n_hidden, n_out)
        self.state_id = 0
        
        self.index = theano.tensor.lscalar('index')
        self.symbolic_gamma_j = theano.tensor.fmatrix('gamma_j')
        self.symbolic_x_j = theano.tensor.fmatrix('x_j')    
        self.train_set_x = theano.tensor.tensor3('x')
        self.gamma = theano.tensor.tensor3('gamma')
        
        # TODO : re-use prob because it has already been computed

        results, updates = theano.scan(lambda v, w: v[:, 0] * \
            theano.tensor.log(self.processOutput(w[0, :])), sequences = [self.gamma, self.train_set_x])
        self.cost = - results.sum()
        
        self.gparams = [theano.tensor.grad(self.cost, param) for param in self.params]
        self.updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, self.gparams)
        ]
        
        self.train_model = theano.function(
            inputs = [self.train_set_x, self.gamma],
            outputs = self.cost,
            updates = self.updates
        )
        
        if not RELEASE_MODE:
            debugfile = open("theano_pistatenetwork_graph.txt", "w")
            theano.printing.debugprint(self.cost, file = debugfile)
            debugfile.close()
    def train(self, train_set_x, gamma, n_epochs = 1):
        train_values_x = np.asarray(train_set_x, dtype = theano.config.floatX)
        gamma_values = np.asarray(gamma, dtype = theano.config.floatX)
        epoch = 0
        while (epoch < n_epochs):
            epoch += 1
            avg_cost = self.train_model(train_values_x, gamma_values)

class StateSubnetwork(MLP):
    def __init__(self, state_id, n_in, n_hidden, n_out, architecture = "ergodic", learning_rate = 0.01):
        MLP.__init__(self, n_in, n_hidden, n_out)
        self.state_id = state_id
        
        self.index = theano.tensor.lscalar('index')
        self.t = theano.tensor.lscalar('t')
        self.symbolic_xi_j = theano.tensor.tensor3('xi_j')
        self.symbolic_x_j = theano.tensor.fmatrix('x_j')
        self.train_set_x = theano.tensor.fmatrix('x')
        self.xi = theano.tensor.tensor3('xi')
        
        # TODO : re-use prob because it has already been computed
        phi = self.processOutput(self.symbolic_x_j[self.t, :])
        self.cost = - (self.symbolic_xi_j[self.state_id, self.t, :] * theano.tensor.log(phi)).sum()
        
        self.gparams = [theano.tensor.grad(self.cost, param) for param in self.params]
        self.updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, self.gparams)
        ]
        
        if not RELEASE_MODE:
            debugfile = open("theano_statenetwork_graph.txt", "w")
            theano.printing.debugprint(self.cost, file = debugfile)
            
        self.train_model = theano.function(
            inputs = [self.symbolic_x_j, self.symbolic_xi_j, self.t],
            outputs = self.cost,
            updates = self.updates
        )
        
    def train(self, train_set_x, xi, n_epochs = 1):
        N = len(train_set_x)
        train_values_x = np.asarray(train_set_x, dtype = theano.config.floatX)
        xi_values = np.asarray(xi, dtype = theano.config.floatX)
        epoch = 0
        while (epoch < n_epochs):
            epoch += 1
            for sequence_id in range(N):
                for j in range(len(train_values_x[sequence_id])):
                    avg_cost = self.train_model(train_values_x[sequence_id], xi_values[sequence_id], j)
        
class OutputSubnetwork(MLP):
    def __init__(self, state_id, n_in, n_hidden, n_out, is_classifier = True, learning_rate = 0.01):
        MLP.__init__(self, n_in, n_hidden, n_out)
        self.state_id = state_id
        self.is_classifier = is_classifier
        self.memory = None
        
        self.index = theano.tensor.lscalar('index')
        self.t = theano.tensor.lscalar('t')
        self.symbolic_memory = theano.tensor.fvector('memory')
        self.symbolic_x_j = theano.tensor.fmatrix('x_j')
        self.train_set_x = theano.tensor.fmatrix('x')
        
        # TODO : re-use prob because it has already been computed
        eta = self.processOutput(self.symbolic_x_j[self.t, :])
        self.cost = - (self.symbolic_memory[self.state_id] * theano.tensor.log(eta)).sum()
        
        self.gparams = [theano.tensor.grad(self.cost, param) for param in self.params]
        self.updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, self.gparams)
        ]
        
        self.train_model = theano.function(
            inputs = [self.symbolic_x_j, self.symbolic_memory, self.t],
            outputs = self.cost,
            updates = self.updates
        )
        
        if not RELEASE_MODE:
            debugfile = open("theano_outputnetwork_graph.txt", "w")
            theano.printing.debugprint(self.cost, file = debugfile)
        
    def train(self, train_set_x, memory_array, n_epochs = 1):
        N = len(train_set_x)
        train_values_x = np.asarray(train_set_x, dtype = theano.config.floatX)
        memory_values = np.asarray(memory_array, dtype = theano.config.floatX)
        
        epoch = 0
        while (epoch < n_epochs):
            epoch += 1
            for sequence_id in range(N):
                for j in range(len(train_values_x[sequence_id])):
                    avg_cost = self.train_model(train_values_x[sequence_id], memory_values[sequence_id], j)

def newStateSubnetworks(n_networks, n_in, n_hidden, n_out, network_type = SUBNETWORK_STATE):
    nets = []
    if network_type == SUBNETWORK_PI_STATE:
        for i in range(n_networks):
            nets.append(PiStateSubnetwork(n_in, n_hidden, n_out))
    elif network_type == SUBNETWORK_STATE:
        for i in range(n_networks):
            nets.append(StateSubnetwork(i, n_in, n_hidden, n_out))
    elif network_type == SUBNETWORK_OUTPUT:
        for i in range(n_networks):
            nets.append(OutputSubnetwork(i, n_in, n_hidden, n_out))
    else:
        raise NotImplementedError()
    return nets

def new3DVLMArray(P, T, ndim = 0, ndim_2 = 0, dtype = np.float):
    if isinstance(ndim, int):
        return np.rand((P, T.max()), dtype = dtype)
    elif ndim_2 == 0:
        if type(ndim) == int:
            return np.empty((P, T.max(), ndim), dtype = dtype)
        else:
            return np.empty((P, T, ndim.max()), dtype = dtype)
    else:
        return np.empty((P, T, ndim.max(), ndim_2), dtype = dtype)

def typedListTo3DPaddedTensor(typed_list, T):
    if isinstance(typed_list, list):
        P = len(T)
        n_max = T.max()
        n_dim = typed_list[0].shape[1]
        new_tensor = np.random.rand(P, n_max, n_dim)
        for p in range(P):
            new_tensor[p, :T[p], :] = typed_list[p][:, :]
        return new_tensor
    else:
        return typed_list
    