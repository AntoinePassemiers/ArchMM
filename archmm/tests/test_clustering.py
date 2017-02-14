# -*- coding: utf-8 -*-

from archmm.tests.utils import *
from archmm.estimation.clustering import *

means = np.array([[-23, 78], [51, -36], [19, 7]])
X1 = (np.random.rand(500, 2) - 0.5) * 4.0 + means[0]
X2 = (np.random.rand(500, 2) - 0.5) * 3.0 + means[1]
X3 = (np.random.rand(500, 2) - 0.5) * 3.5 + means[2]
X = np.concatenate((X1, X2, X3), axis = 0)

def test_compare_algorithms():
	cmeans = fuzzyCMeans(X, 3)[0]
	assert_array_almost_equal_by_permuting(cmeans, means, decimal = 1, axis = 0)