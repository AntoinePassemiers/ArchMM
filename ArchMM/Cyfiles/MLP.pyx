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

class HiddenLayer(Layer):
    def __init__(self, inputv, n_in, n_out, rng, activation):
        self.input = inputv
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        self.activation = activation
        
        W_array = np.asarray(
            rng.uniform(
                low = -np.sqrt(6.0 / (self.n_in + self.n_out)),
                high = np.sqrt(6.0 / (self.n_in + self.n_out))
                ),
            dtype = theano.config.floatX
        )
        if activation == T.nnet.sigmoid:
            W_array *= 4
        self.W = theano.shared(value = W_array, name = "W", borrow = True)
        b_array = np.ones((n_out,), dtype = theano.config.floatX)
        self.b = theano.shared(value = b_array, name = "b", borrow = True)
        temp = T.dot(inputv, self.W) + self.b
        self.output = temp if not activation else activation(temp)
        self.params = [self.W, self.b]
      
class OutputLayer(Layer):
    def __init__(self, inputv, n_in, n_out, rng):
        W_array = np.asarray(np.random.randn(n_in, n_out), dtype = theano.config.floatX)
        b_array = np.asarray(np.ones(n_out), dtype = theano.config.floatX)
        self.W = theano.shared(value = W_array, name = "W", borrow = True)
        self.b = theano.shared(value = b_array, name = "b", borrow = True)
        self.p_y_given_x = T.nnet.softmax(T.dot(inputv, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        self.params = [self.W, self.b]
        self.input = inputv
        self.activation = T.nnet.softmax
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                            ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            return NotImplementedError()
      
        
class MLP:
    def __init__(self, n_in, n_hidden, n_out, rng = np.random.RandomState(4), 
                 learning_rate = 0.01, L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 1000):
        self.X = T.matrix('X')
        self.y = T.ivector('y')
        self.input = self.X
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.rng = rng
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.n_epochs = n_epochs
        
        self.hidden_layer = HiddenLayer(self.X, n_in, n_hidden, rng, T.tanh)
        self.output_layer = OutputLayer(self.hidden_layer.output, n_hidden, n_out, rng)
        self.params = self.hidden_layer.params + self.output_layer.params
        self.layers = [self.hidden_layer, self.output_layer]
        
        self.L1 = np.abs(self.hidden_layer.W).sum() + np.abs(self.output_layer.W).sum()
        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.output_layer.W ** 2).sum()

        self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        self.errors = self.output_layer.errors
        self.cost = self.negative_log_likelihood(self.y) + self.L1_reg * self.L1 + self.L2_reg + self.L2_sqr 
        self.gradient_params = [T.grad(self.cost, param) for param in self.params]
        self.updates = [
            (param, param - self.learning_rate * gradient_param)
            for param, gradient_param in zip(self.params, self.gradient_params)
        ]
        self.predicter = theano.function([self.X], self.processOutput(self.X))
    
    def train(self, X_array, y_array):
        # y_array must contain only 0 and 1
        train_X = theano.shared(value = X_array, name = "train_X", borrow = True)
        train_y = theano.shared(value = y_array, name = "train_y", borrow = True)
        index = T.lscalar()
        self.trainer = theano.function(
            inputs = [index],
            outputs = self.cost,
            updates = self.updates,
            givens = {
                self.X : train_X[index : (index + 1)],
                self.y : train_y[index : (index + 1)] 
            }
        )
        
        start_time = timeit.default_timer()
        epoch = 0
        while epoch < self.n_epochs:
            epoch += 1
            for i in range(train_X.shape[0].eval()):
                avg_cost = self.trainer(i)
        total_time = timeit.default_timer() - start_time
        print("Total training time : %i min %i" % (total_time / 60, total_time % 60))
        
    def processOutput(self, X):
        for layer in self.layers:
            X = layer.processOutput(X)
        return X
        
    def predict(self, X_array):
        test_X = theano.shared(value = X_array, name = "test_X", borrow = True)
        return self.processOutput(test_X)
    

class StateSubnetwork(MLP):
    pass
class OutputSubnetwork(MLP):
    pass

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

    