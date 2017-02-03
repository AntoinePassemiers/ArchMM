# -*- coding: utf-8 -*-

import numpy as np

import theano


class BinarySVM:
    def __init__(self, state_id, n_in):
        self.state_id = state_id
        self.n_in = n_in
        self.y = theano.tensor.vector(name = 'y', dtype = theano.tensor.floatX)
        self.X = theano.tensor.matrix(name = "X", dtype = theano.tensor.floatX)
        self.W = theano.tensor.vector(name = "W", dtype = theano.tensor.floatX)
        self.lambda_par = theano.tensor.fscalar(name = "lamda") # Regularization parameter
        self.m = theano.tensor.fscalar(name = "m") # Number of examples
        self.t = theano.tensor.lscalar('t')
        self.s = theano.tensor.lscalar('s')
        self.symbolic_memory = theano.tensor.tensor3(name = 'memory', dtype = theano.config.floatX)
        self.symbolic_x = theano.tensor.tensor3(name = 'x_j', dtype = theano.config.floatX)
        self.train_set_x = theano.tensor.matrix(name = 'x', dtype = theano.config.floatX)
        self.learning_rate = theano.tensor.fscalar("learning_rate")
        self.symbolic_target = theano.tensor.imatrix(name = "target")
        
        eta = theano.tensor.switch(theano.tensor.dot(self.symbolic_x[self.s, self.t, :], self.W) > 0.0, 1.0, 0.0)
        iohmm_cost = - (self.symbolic_memory[self.s, self.t, self.state_id] * \
                       theano.tensor.log(eta[self.symbolic_target[self.s, self.t]])).sum()
        self.cost = (lambda_par / 2.0) * theano.pow(W).sum() + iohmm_cost
        
        self.params = [self.W]
        self.gparams = [theano.tensor.grad(self.cost, param) for param in self.params]
        self.updates = list()
        for param, gparam in zip(self.params, self.gparams):
            weight_update = param - self.learning_rate * gparam
            self.updates.append((param, weight_update))
        self.train_model = theano.function(
            inputs = [self.X, self.y, self.symbolic_memory, self.s, self.t, self.learning_rate],
            outputs = self.cost,
            updates = self.updates
        )
        
        self.computeOutputFunction = theano.function(inputs = [symbolic_X_j_k], outputs = next_layer_input)
        
    def computeOutput(self, X):
        return self.computeOutputFunction(X)
    
    def processOutput(self, X):
        raise NotImplementedError()
    
    def __getstate__(self):
        pass
    
    def __setstate__(self, state):
        pass
    
    def train(self, train_set_x, target_set, memory_array, weights, is_mv, n_epochs = 1, learning_rate = 0.05):
        N = len(train_set_x)
        T = len(train_set_x[0]) # Warning : Sequence length is supposed to be constant
        assert(N == len(target_set))
        epoch = 0
        while (epoch < n_epochs):
            epoch += 1
            M = 0
            avg_cost = 0
            for j in range(T):
                for sequence_id in range(N):
                    if not is_mv[sequence_id, j]:
                        if weights[sequence_id] != 0:
                            cost = self.train_model(train_set_x, target_set,
                                    memory_array, sequence_id, j, float(learning_rate * weights[sequence_id]))
                            avg_cost += cost
                            M += 1
        return avg_cost / float(M)