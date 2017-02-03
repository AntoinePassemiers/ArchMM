# -*- coding: utf-8 -*-

import numpy as np
import theano, theano.tensor

theano.config.floatX = 'float32'
theano.config.allow_gc = True
theano.config.scan.allow_gc = True
theano.config.scan.allow_output_prealloc = False


class MLLR:
    def __init__(self):
        gamma   = theano.tensor.matrix(name = "gamma", dtype = theano.config.floatX)
        mu_hat  = theano.tensor.matrix(name = "mu", dtype = theano.config.floatX)
        sigma_T = theano.tensor.tensor3(name = "sigma_T", dtype = theano.config.floatX)
        X = theano.tensor.matrix(name = "X", dtype = theano.config.floatX)
        M = theano.tensor.matrix(name = "M", dtype = theano.config.floatX)
        b = theano.tensor.vector(name = "b", dtype = theano.config.floatX)
        
        s = theano.tensor.iscalar(name = "s") # Current hidden state
        t = theano.tensor.iscalar(name = "t") # Current time
        
        distance_to_centroid = X[t, :] - mu_hat[s, :]
        mahalanobis_distance = distance_to_centroid.T * sigma_T[s, :, :] * distance_to_centroid
        cost_at_time_t = 0.5 * gamma[s, t] * mahalanobis_distance
        
        """ TODO
        
        - Loop over t using a theano function
        - Compute the theano.grad of W and b
        - Create a theano function for the gradient descent
        """
        
        
def testMLLR():
    MLLR()
    
if __name__ == "__main__":
    testMLLR()
    print("Finished")