# -*- coding: utf-8 -*-

import numpy as np
import theano
from theano.tensor.signal import pool

from archmm.ANN.Layers import *

# http://deeplearning.net/tutorial/code/convolutional_mlp.py

class Conv2DLayer(Layer):
    def __init__(self, input, height, width, pool_shp = (2, 2), W = None, b = None,
                 activation = theano.tensor.tanh,
                 rng = np.random.RandomState(1234)):
        self.is_conv = True
        self.input = input
        self.activation = activation
        self.kernel_height = height
        self.kernel_width  = width
        self.pool_shp = pool_shp
        filter_shape = (1, 1, width, height)
        W_bound = np.sqrt(height * width)
        
        n_in = np.prod(filter_shape[1:])
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(pool_shp))
        
        if W is None:
            W_values = np.asarray(
                rng.uniform(low = -W_bound, high = W_bound, size = filter_shape),
                dtype = theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value = W_values, name = 'W', borrow = True)

        if b is None:
            b_values = np.asarray(rng.uniform(low = -0.5, high = 0.5, 
                        size = (filter_shape[0],)), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        
        self.W = W
        self.b = b

        conv_out = theano.tensor.nnet.conv2d(input, self.W)
        pooled_out = pool.pool_2d(
            input = conv_out,
            ds = pool_shp,
            ignore_border = True
        )
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = (
            lin_output if activation is None
            else activation(lin_output.flatten(ndim = 1))
        )
        self.params = [self.W, self.b]