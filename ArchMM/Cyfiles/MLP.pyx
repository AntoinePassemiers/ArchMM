# -*- coding: utf-8 -*-

import numpy as np
import os, timeit

from Utils import *

os.environ["THEANO_FLAGS"] = "floatX=float32,exception_verbosity=high" # TODO : doesn't work

import theano
import theano.typed_list

SUBNETWORK_STATE = 301
SUBNETWORK_OUTPUT = 302

class Layer:  
    def processOutput(self, X):
        linear_output = theano.tensor.dot(X, self.W) + self.b
        if self.activation is not None:
            output = self.activation(linear_output)
        else:
            output = linear_output
        return output

class LogisticRegression(Layer):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
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
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
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
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
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
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
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
    


class StateSubnetwork(MLP):
    def __init__(self, state_id, n_in, n_hidden, n_out):
        MLP.__init__(self, n_in, n_hidden, n_out)
        self.state_id = state_id
        self.xi = None
    def train(self, train_list_x, xi_list, learning_rate = 0.01, 
              L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 1):
        
        N = len(train_list_x)
        m = train_list_x[0].shape[1]
        train_set_x = theano.typed_list.TypedListType(theano.tensor.fmatrix)()
        xi = theano.typed_list.TypedListType(theano.tensor.ftensor3)()
        for i in range(N):
            xi_values = np.asarray(xi_list[i], dtype = theano.config.floatX)
            X_values = np.asarray(train_list_x[i], dtype = theano.config.floatX)
            theano.typed_list.append(train_set_x, theano.shared(name = "X_train", borrow = True, value = X_values))
            theano.typed_list.append(xi, theano.shared(name = "xi", borrow = True, value = xi_values))
        self.xi = theano.tensor.tensor3('xi_j')
        index = theano.tensor.lscalar()
        self.j = theano.tensor.lscalar()
        j_t = theano.tensor.lscalar()
        
        # prob = self.processOutput(self.x[self.j + 1, :])
        cost = - (self.xi[self.state_id, :, self.j] * theano.tensor.log(45)).sum()

        gparams = [theano.tensor.grad(cost, param) for param in self.params]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        
        train_model = theano.function(
            inputs = [index, self.j],
            outputs = cost,
            updates = updates,
            givens = {
                self.x: train_set_x[index],
                self.xi: xi[index]
            }
        )
        if not RELEASE_MODE:
            theano.printing.debugprint(cost)
        
        epoch = 0
        while (epoch < n_epochs):
            epoch += 1
            for sequence_id in range(N):
                for j in range(len(train_list_x[sequence_id])):
                    minibatch_avg_cost = train_model(sequence_id, j)
        
class OutputSubnetwork(MLP):
    def __init__(self, output_id, n_in, n_hidden, n_out):
        MLP.__init__(self, n_in, n_hidden, n_out)
        self.output_id = output_id
    def train(self, train_set_x, train_set_y,
                 learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
                 batch_size=20):
        X_values = np.asarray(train_set_x, dtype = theano.config.floatX)
        y_values = np.asarray(train_set_y, dtype = np.int)
        train_set_x = theano.shared(name = "X_train", borrow = True, value = X_values)
        train_set_y = theano.shared(name = "y_train", borrow = True, value = y_values)
        index = theano.tensor.lscalar()
        y = theano.tensor.ivector('y')
        
        cost = (
            self.negative_log_likelihood(y)
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        gparams = [theano.tensor.grad(cost, param) for param in self.params]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        
        b = train_set_x.shape[0].eval()
        if batch_size > b:
            batch_size = b
            b = 1
        else:
            b /= batch_size
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        improvement_threshold = 0.995
        epoch = 0
        while (epoch < n_epochs):
            epoch = epoch + 1
            for minibatch in range(b):
                minibatch_avg_cost = train_model(minibatch)

def newStateSubnetworks(n_networks, n_in, n_hidden, n_out, network_type = SUBNETWORK_STATE):
    nets = []
    if network_type == SUBNETWORK_STATE:
        for i in range(n_networks):
            nets.append(StateSubnetwork(i, n_in, n_hidden, n_out))
    elif network_type == SUBNETWORK_OUTPUT:
        for i in range(n_networks):
            nets.append(OutputSubnetwork(i, n_in, n_hidden, n_out))
    else:
        raise NotImplementedError()
    return nets

    