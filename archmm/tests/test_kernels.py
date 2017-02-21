# -*- coding: utf-8 -*-

import numpy as np

from archmm.svm.kernel import *
from archmm.tests.utils import *
from archmm.utils import *


@todo
def setup_func():
    pass

@todo
def teardown_func():
    pass

@with_setup(setup_func, teardown_func)
def test_kernel_matrix_dims():
    X = np.random.rand(2750, 57)
    assert_equals(X.shape == linear_kernel(X).shape)
    assert_equals(X.shape == polynomial_kernel(X).shape)
    X = np.random.rand(23, 900)
    assert_equals(X.shape == linear_kernel(X).shape)
    assert_equals(X.shape == polynomial_kernel(X).shape)