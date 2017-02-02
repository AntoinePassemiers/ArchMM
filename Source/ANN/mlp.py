# -*- coding: utf-8 -*-

from archmm.ann.layers import *
from archmm.ann.cnn import *

RELEASE_MODE = False

SUBNETWORK_PI_STATE = 300
SUBNETWORK_STATE    = 301
SUBNETWORK_OUTPUT   = 302

ERGODIC_LAYER = 401
LINEAR_LAYER  = 402

DEBUG_MODE = theano.compile.MonitorMode(
    post_func = theano.compile.monitormode.detect_nan).excluding('local_elemwise_fusion', 'inplace')
    
def expsum(x):
    return theano.tensor.sum(theano.tensor.exp(x))

class MLP(object):
    def __init__(self, n_in, n_hidden, n_out, rng = np.random.RandomState(1234), dropout_threshold = 0.5,
                 hidden_activation_function = "sigmoid"):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.input = self.x = theano.tensor.matrix(name = 'x', dtype = theano.config.floatX)
        self.rng = rng
        self.dropout_threshold = dropout_threshold
        srng = RandomStreams(seed = 1234)
        self.rnd_number = srng.uniform(ndim = 0, low = 0.0, high = 1.0)
        if hidden_activation_function == "tanh":
            self.activation = theano.tensor.tanh
        elif hidden_activation_function == "relu":
            self.activation = theano.tensor.nnet.relu
        else:
            self.activation =  theano.tensor.nnet.nnet.sigmoid
        
        self.hiddenLayer = HiddenLayer(
            self.input, n_in, n_hidden, 
            rng = self.rng, activation = self.activation
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
        
        symbolic_X_j_k = theano.tensor.vector(name = "X", dtype = theano.config.floatX)
        next_layer_input = symbolic_X_j_k
        i = 0 
        for layer in self.layers:
            if i == 1:
                next_layer_input = theano.tensor.switch(self.rnd_number < 0.1, 0, next_layer_input)
            next_layer_input = layer.processOutput(next_layer_input)
            i += 1
        
        self.computeOutputFunction = theano.function(inputs = [symbolic_X_j_k], outputs = next_layer_input)
        
        self.dropout_or_nan_to_num = theano.tensor.switch(self.rnd_number < dropout_threshold, 0.00001, 0)

    def computeOutput(self, X):
        return self.computeOutputFunction(X)

    def processOutput(self, X):
        for layer in self.layers:
            X = layer.processOutput(X)
        return X # theano.tensor.switch(theano.tensor.isnan(X), 0.0001, X)
    
    def __getstate__(self):
        state = list()
        for layer in self.layers:
            state.append(layer.__getstate__())
        return state
    
    def __setstate__(self, state):
        i = 0
        for layer in self.layers:
            W, b = state[i]
            layer.__setstate__(state[i])
            i += 1
    
class PiStateSubnetwork(MLP):
    def __init__(self, n_in, n_hidden, n_out, dropout_threshold = 0.5,
                 hidden_activation_function = "sigmoid", learning_rate = 0.01, architecture = "ergodic"):
        assert(architecture in ["ergodic", "linear"])
        # self.architecture = ERGODIC_LAYER if architecture == "ergodic" else LINEAR_LAYER
        self.architecture = ERGODIC_LAYER
        MLP.__init__(self, n_in, n_hidden, n_out, 
                     hidden_activation_function = hidden_activation_function,
                     dropout_threshold = dropout_threshold)
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.state_id = 0
        self.index = theano.tensor.lscalar('index')
        self.symbolic_gamma_j = theano.tensor.matrix(name = 'gamma_j', dtype = theano.config.floatX)
        self.train_set_x = theano.tensor.tensor3(name = 'x', dtype = theano.config.floatX)
        self.gamma = theano.tensor.tensor3(name = 'gamma', dtype = theano.config.floatX)
        self.learning_rate = theano.tensor.scalar("learning_rate", dtype = theano.config.floatX)
        self.s = theano.tensor.lscalar('s')
        self.cost = - theano.tensor.dot(self.gamma[self.s, :, 0],
            theano.tensor.log(self.processOutput(self.train_set_x[self.s, 0, :])[0]))
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

    def train(self, train_set_x, gamma, weights, n_epochs = 1, learning_rate = 0.01):
        if self.architecture == ERGODIC_LAYER:
            epoch = 0
            N = len(train_set_x)
            while (epoch < n_epochs):
                epoch += 1
                avg_cost = 0
                for sequence_id in range(N):
                    if weights[sequence_id] != 0:
                        cost = self.train_model(train_set_x, 
                            gamma, sequence_id, float(learning_rate * weights[sequence_id]))
                        avg_cost += cost
                avg_cost /= N
            return avg_cost
        else:
            return 0.0
    
    def computeOutput(self, X):
        if self.architecture == ERGODIC_LAYER:
            return MLP.computeOutput(self, X)
        else:
            output = np.zeros(self.n_out, dtype = theano.config.floatX)
            output[0] = 1.0
            return np.array([output])
            
    def processOutput(self, X):
        if self.architecture == ERGODIC_LAYER:
            return MLP.processOutput(self, X)
        else:
            output = np.zeros(self.n_out, dtype = theano.config.floatX)
            output[0] = 1.0
            return theano.tensor.switch(self.rnd_number > 0.0, np.array([output]), MLP.processOutput(self, X))

class StateSubnetwork(MLP):
    def __init__(self, state_id, n_in, n_hidden, n_out, architecture = "ergodic", 
                 dropout_threshold = 0.5,
                 hidden_activation_function = "sigmoid", learning_rate = 0.01):
        assert(architecture in ["ergodic", "linear"])
        self.architecture = ERGODIC_LAYER if architecture == "ergodic" else LINEAR_LAYER
        n_out_to_compute = n_out if self.architecture == ERGODIC_LAYER else 2
        MLP.__init__(self, n_in, n_hidden, n_out_to_compute, 
                     hidden_activation_function = hidden_activation_function,
                     dropout_threshold = dropout_threshold)
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.state_id = state_id
        self.index = theano.tensor.lscalar('index')
        self.t = theano.tensor.lscalar('t')
        self.s = theano.tensor.lscalar('s')
        self.symbolic_xi = theano.tensor.tensor4(name = 'xi', dtype = theano.config.floatX)
        self.symbolic_x = theano.tensor.tensor3(name = 'x', dtype = theano.config.floatX)
        self.learning_rate = theano.tensor.fscalar("learning_rate")
        
        phi = self.processOutput(self.symbolic_x[self.s, self.t, :])[0]
        self.cost = - (self.symbolic_xi[self.s, self.state_id, self.t, :] * theano.tensor.log(phi)).sum()
        self.gparams = [theano.tensor.grad(self.cost, param) for param in self.params]
        self.updates = list()
        for param, gparam in zip(self.params, self.gparams):
            weight_update = param - self.learning_rate * gparam
            self.updates.append((param, theano.tensor.switch(theano.tensor.isnan(weight_update), 
                                self.dropout_or_nan_to_num, weight_update)))
        self.train_model = theano.function(
            inputs = [self.symbolic_x, self.symbolic_xi, self.s, self.t, self.learning_rate],
            outputs = self.cost,
            updates = self.updates
        )
        
        phi = self.processOutput(self.symbolic_x[self.s, self.t, :])[0]
        self.log_cost = - expsum(self.symbolic_xi[self.s, self.state_id, self.t, :] + theano.tensor.log(phi))
        self.log_gparams = [theano.tensor.grad(self.log_cost, param) for param in self.params]
        self.log_updates = list()
        for param, gparam in zip(self.params, self.log_gparams):
            weight_update = param - self.learning_rate * gparam
            self.log_updates.append((param, theano.tensor.switch(theano.tensor.isnan(weight_update), 
                                self.dropout_or_nan_to_num, weight_update)))
        self.train_log_model = theano.function(
            inputs = [self.symbolic_x, self.symbolic_xi, self.s, self.t, self.learning_rate],
            outputs = self.log_cost,
            updates = self.log_updates
        )
        
    def train(self, train_set_x, xi, weights, is_mv, n_epochs = 1, learning_rate = 0.01):
        if not (self.architecture == LINEAR_LAYER and self.state_id == self.n_out - 1):
            N = len(train_set_x)
            T = len(train_set_x[0]) # Warning : Sequence length is supposed to be constant
            epoch = 0
            avg_cost = 0
            while (epoch < n_epochs):
                epoch += 1
                M = 0
                avg_cost = 0
                for j in range(T):
                    for sequence_id in range(N):
                        if not is_mv[sequence_id, j]:
                            if weights[sequence_id] != 0:
                                cost = self.train_model(train_set_x, xi, sequence_id, j, float(learning_rate * weights[sequence_id]))
                                avg_cost += cost
                            M += 1
            return avg_cost / float(M)
        else:
            return 0.0

    def computeOutput(self, X):
        if self.architecture == ERGODIC_LAYER:
            return MLP.computeOutput(self, X)
        else:
            if self.state_id >= self.n_out - 1:
                output = np.zeros(self.n_out, dtype = theano.config.floatX)
                output[-1] = 1.0
                return np.array([output])
            else:
                begin = np.zeros(self.state_id, dtype = theano.config.floatX)
                output = MLP.computeOutput(self, X)[0]
                end = np.zeros(self.n_out - 2 - self.state_id, dtype = theano.config.floatX)
                return np.array([np.concatenate([begin, output, end])])
           
    def processOutput(self, X):
        if self.architecture == ERGODIC_LAYER:
            return MLP.processOutput(self, X)
        else:
            if self.state_id >= self.n_out - 1:
                return MLP.processOutput(self, X)
            else:
                begin = np.zeros(self.state_id, dtype = theano.config.floatX)
                output = MLP.processOutput(self, X)[0]
                end = np.zeros(self.n_out - 2 + self.state_id, dtype = theano.config.floatX)
                return theano.tensor.concatenate([begin, output, end])

        
class OutputSubnetwork(MLP):
    def __init__(self, state_id, n_in, n_hidden, n_out, is_classifier = True,
                 dropout_threshold = 0.5, 
                 hidden_activation_function = "sigmoid", learning_rate = 0.01):
        MLP.__init__(self, n_in, n_hidden, n_out, 
                     hidden_activation_function = hidden_activation_function,
                     dropout_threshold = dropout_threshold)
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.state_id = state_id
        self.is_classifier = is_classifier
        self.memory = None
        self.index = theano.tensor.lscalar('index')
        self.t = theano.tensor.lscalar('t')
        self.s = theano.tensor.lscalar('s')
        self.symbolic_memory = theano.tensor.tensor3(name = 'memory', dtype = theano.config.floatX)
        self.symbolic_x = theano.tensor.tensor3(name = 'x_j', dtype = theano.config.floatX)
        self.train_set_x = theano.tensor.matrix(name = 'x', dtype = theano.config.floatX)
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
    
class Supervisor:
    def __init__(self, targets, n_iterations, n_examples, n_states):
        self.n_examples = n_examples
        self.n_states = n_states
        self.n_iterations = n_iterations
        self.history = np.empty((n_iterations + 1, n_examples), dtype = np.float32)
        self.history[0, :] = - np.inf
        self.weights = np.ones((n_iterations + 1, n_examples), dtype = np.float32)
        self.labels = np.empty(n_examples, dtype = np.int32)
        for i in range(n_examples):
            self.labels[i] = targets[i, -1]
        self.current_iter = 0
        self.hardworker = 0
    def next(self, loglikelihoods, pi_state_costs, state_costs, output_costs):
        self.hardworker = output_costs.argmax()
        self.current_iter += 1
        self.history[self.current_iter] = loglikelihoods
        signs = np.sign(loglikelihoods - self.history[self.current_iter - 1])
        signs[np.isnan(signs)] = 0
        # indexes = (signs <= 0)
        indexes = np.less(loglikelihoods, 0)
        decay = float(self.n_iterations - self.current_iter) / float(self.n_iterations)
        self.weights[self.current_iter, indexes] = 3 * decay * np.sqrt(self.weights[self.current_iter - 1, indexes]) + 1
        self.weights[self.current_iter, loglikelihoods > 0] = 0
        print(loglikelihoods, self.weights[self.current_iter])
        t_weights = np.ones(2, dtype = np.float32)
        o_weights = np.ones(self.n_states, dtype = np.float32)
        # o_weights[self.hardworker] = 1.0
        return self.weights[self.current_iter], t_weights, o_weights

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

    