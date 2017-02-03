# -*- coding: utf-8 -*-

from archmm.tests.utils import *
from archmm.clustering import *


X1 = (np.random.rand(500, 2) - 0.5) * 4.0 + np.array([-23, 78])
X2 = (np.random.rand(500, 2) - 0.5) * 3.0 + np.array([51, -36])
X3 = (np.random.rand(500, 2) - 0.5) * 3.5 + np.array([19, 7])
X = np.concatenate((X1, X2, X3), axis = 0)

def test_compare_algorithms():
	cc = fuzzyCMeans(X, 3)[0]
	ck = kMeans(X, 3)

	print(cc)
	print(ck)