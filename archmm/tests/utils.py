# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import *


def find_nearest(arr, value, axis = 1):
	return np.linalg.norm(np.abs(arr - value), axis = axis).argmin()

def assert_array_almost_equal_by_permuting(A, B, axis = 0, **kwargs):
	""" Only avaiable for 2d arrays """
	permuted_A = np.empty(A.shape)
	for i in range(len(B)):
		permuted_A[i] = A[find_nearest(A, B[i], axis = (1 - axis))]
	assert_array_almost_equal(permuted_A, B, **kwargs)
