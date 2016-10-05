# -*- coding: utf-8 -*-

import numpy as np
import timeit
import theano
import theano.tensor as T

# https://eldorado.tu-dortmund.de/bitstream/2003/5496/1/03010.pdf

SUBNETWORK_STATE = 301
SUBNETWORK_OUTPUT = 302

class Layer:  
    def processOutput(self, X):
        linear_output = T.dot(X, self.W) + self.b
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
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input
        self.activation = T.nnet.softmax
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
        
class HiddenLayer(Layer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.nnet.sigmoid):
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
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, n_in, n_hidden, n_out, rng = np.random.RandomState(1234)):
        self.input = self.x = T.matrix('x')
        self.rng = rng
        self.hiddenLayer = HiddenLayer(
            rng=self.rng,
            input=self.input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
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
    
    def train(self, train_set_x, train_set_y,
                 learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
                 batch_size=20, n_hidden=500):
        X_values = np.asarray(train_set_x, dtype = theano.config.floatX)
        y_values = np.asarray(train_set_y, dtype = np.int)
        train_set_x = theano.shared(name = "X_train", borrow = True, value = X_values)
        train_set_y = theano.shared(name = "y_train", borrow = True, value = y_values)
        index = T.lscalar()
        y = T.ivector('y')
        cost = (
            self.negative_log_likelihood(y)
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        gparams = [T.grad(cost, param) for param in self.params]
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
    
    def predict(self, test_X):
        X_values = np.asarray(test_X, dtype = theano.config.floatX)
        test_X = theano.shared(name = "X_test", borrow = True, value = X_values)
        return self.processOutput(test_X).eval() 

class StateSubnetwork(MLP):
    def updateParameters(self, X, alpha, beta):
        pred_Y = self.predict(X)
        print(pred_Y.shape)
        
class OutputSubnetwork(MLP):
    def updateParameters(self, X, zeta, B):
        pred_Y = self.predict(X)

def newStateSubnetworks(n_networks, n_in, n_hidden, n_out, network_type = SUBNETWORK_STATE):
    nets = []
    if network_type == SUBNETWORK_STATE:
        for i in range(n_networks):
            nets.append(StateSubnetwork(n_in, n_hidden, n_out))
    elif network_type == SUBNETWORK_OUTPUT:
        for i in range(n_networks):
            nets.append(OutputSubnetwork(n_in, n_hidden, n_out))
    else:
        raise NotImplementedError()
    return nets

    